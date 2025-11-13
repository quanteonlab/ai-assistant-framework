# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 14)

**Starting Chapter:** What If We Do Not Have Labeled Data

---

#### Zero-shot Classification Overview
Background context: Zero-shot classification is a method used when labeled data for training a classifier is unavailable. Instead, we use descriptions of labels and embeddings to classify documents without explicit label training.

:p What is zero-shot classification?
??x
Zero-shot classification is a technique where the model predicts labels that were not seen during training because the dataset does not contain any examples with those specific labels. The process involves embedding both document texts and label descriptions, then using cosine similarity to determine the best match.
x??

---

#### Creating Label Embeddings
Background context: To perform zero-shot classification, we first need to create embeddings for our labels based on their descriptions.

:p How do you create label embeddings?
??x
To create label embeddings, you can use a model's `.encode` function with the label descriptions. For example:

```python
label_embeddings = model.encode(["A negative review", "A positive review"])
```

This step transforms text into numerical vectors that can be used for comparison.
x??

---

#### Calculating Cosine Similarity
Background context: Cosine similarity is a measure of the angle between two vectors, which helps in determining how similar the document embeddings are to label embeddings.

:p What is cosine similarity and how do you calculate it?
??x
Cosine similarity measures the cosine of the angle between two non-zero vectors. It is calculated as:

$$\text{cosine\_similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\|_2 \|\mathbf{B}\|_2}$$

Where:
- $\mathbf{A} \cdot \mathbf{B}$ is the dot product of vectors A and B.
- $\|\mathbf{A}\|_2 $ and$\|\mathbf{B}\|_2$ are the L2 norms (lengths) of vectors A and B.

In Python, you can calculate cosine similarity using:

```python
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(test_embeddings, label_embeddings)
```

This code computes a matrix where each element is the cosine similarity between a document embedding and a label embedding.
x??

---

#### Predicting Labels Using Cosine Similarity
Background context: After creating embeddings for documents and labels, we use cosine similarity to determine the most similar label for each document.

:p How do you predict labels using cosine similarity?
??x
To predict labels, you can find the maximum cosine similarity score between each document embedding and all label embeddings:

```python
from sklearn.metrics.pairwise import cosine_similarity

# Compute the similarity matrix
sim_matrix = cosine_similarity(test_embeddings, label_embeddings)

# Predict the label with the highest similarity for each document
y_pred = np.argmax(sim_matrix, axis=1)
```

This logic selects the label that has the highest cosine similarity score as the predicted label for each document.
x??

---

#### Evaluating Performance
Background context: After predicting labels using zero-shot classification, it is crucial to evaluate the model's performance. This typically involves comparing the predicted labels with actual labels.

:p How do you evaluate the performance of a zero-shot classifier?
??x
To evaluate the performance, you can use metrics such as precision, recall, and F1-score. The `evaluate_performance` function can be called to get these metrics:

```python
evaluate_performance(data["test"]["label"], y_pred)
```

This will provide detailed metrics on how well the model performed.
x??

---

#### Impact of Label Descriptions on Results
Background context: The choice of label descriptions can significantly impact the performance of zero-shot classification. More specific and relevant descriptions may yield better results.

:p How does choosing more specific label descriptions affect zero-shot classification?
??x
Choosing more specific label descriptions can improve the model's ability to understand the context and focus on relevant features. For example, using "A very negative movie review" instead of just "Negative Review" might make the embeddings capture more nuances related to the context.

This improvement comes from better alignment between the document content and the label semantics.
x??

---

#### Text Classification with Generative Models Overview
Generative models, such as OpenAI’s GPT and Google's T5, are sequence-to-sequence models that generate new sequences of tokens based on input. Unlike task-specific models which output a class directly, these generative models require additional guidance to understand the context and perform specific tasks.
:p What is the primary difference between generative models like GPT and task-specific models?
??x
Generative models like GPT require explicit instructions or prompts to perform specific tasks, whereas task-specific models such as BERT can output a class directly without needing explicit instructions. This means that for text classification using GPT-like models, you need to provide a guiding prompt.
x??

---

#### T5 Architecture and Pretraining
The Text-to-Text Transfer Transformer (T5) model uses an encoder-decoder architecture similar to the original Transformer. During pretraining, it predicts masked token spans rather than individual tokens. This allows the model to handle longer sequences better.
:p How does the pretraining step of T5 differ from that of a typical language model?
??x
In T5's pretraining phase, sets of tokens (token spans) are masked and the model has to predict these masks, unlike in traditional masked language models where individual tokens are masked. This allows the model to process longer sequences more effectively.
x??

---

#### Fine-Tuning T5 for Text Classification
During fine-tuning, the T5 model is trained on a wide variety of tasks by converting them into sequence-to-sequence problems. The Flan-T5 family of models benefits from this extensive training across numerous tasks.
:p How does fine-tuning work in the context of using T5 for text classification?
??x
Fine-tuning involves converting specific tasks (like text classification) into textual instructions and training the model on these instructions simultaneously with other tasks. This allows the model to learn a broad range of language understanding and generation abilities, making it more versatile.
x??

---

#### Using Flan-T5 for Text Classification
To use Flan-T5 for text classification, you need to load the model using the `text2text-generation` task and provide prompts that guide the model towards the desired output. The prompts help the model understand the context of the input.
:p How do you prepare data for a T5 model in a text classification task?
??x
You prefix each document with a prompt, such as "Is the following sentence positive or negative?" to guide the model. Then, map this prompt to your dataset examples and run the pipeline to generate predictions based on these prompts.
```python
# Example code snippet
prompt = "Is the following sentence positive or negative? "
data = data.map(lambda example: {"t5": prompt + example['text']})
```
x??

---

#### Evaluating Model Performance
After generating text, you need to convert the textual output into numerical values (e.g., 0 for 'negative', 1 for 'positive') and evaluate the model's performance using metrics like precision, recall, and F1-score.
:p How do you evaluate the performance of a T5 model after text generation?
??x
You map the textual predictions to numerical labels, calculate precision, recall, and F1-score, and assess the model's overall accuracy. For instance:
```python
y_pred = []
for output in tqdm(pipe(KeyDataset(data["test"], "t5")), total=len(data["test"])):
    text = output[0]["generated_text"]
    y_pred.append(0 if text == "negative" else 1)
evaluate_performance(data["test"]["label"], y_pred)
```
x??

---

#### Background on ChatGPT and its Training Process
Background context: The provided text discusses the training process of ChatGPT, a closed-source model based on the decoder-only architecture. OpenAI used preference tuning to fine-tune the initial model into ChatGPT by manually creating desired outputs, ranking them, and using this data for final adjustments.
:p What is the primary method OpenAI used to fine-tune the ChatGPT model?
??x
OpenAI used preference tuning by manually creating desired outputs (instruction data) and then ranking multiple generated outputs from best to worst. This ranking was used to create the final model, ChatGPT.
x??

---

#### Accessing ChatGPT via API
Background context: To use a closed-source model like ChatGPT in your application, you need to access it through OpenAI’s API after creating an API key. The example provided shows how to set up and use this API for generating text based on specific prompts.
:p How do you create a client object to interact with the OpenAI server using Python?
??x
To create a client object that interacts with OpenAI's servers, you need to install the `openai` package and then initialize it with your API key. Here is how you can do it:

```python
import openai

# Create client
client = openai.OpenAI(api_key="YOUR_API_KEY_HERE")
```
x??

---

#### Generating Text Using ChatGPT
Background context: The text explains the `chatgpt_generation` function, which allows generating text based on a prompt and input document. It uses OpenAI’s API to communicate with the server.
:p How is the `chatgpt_generation` function defined in the provided example?
??x
The `chatgpt_generation` function generates output based on a given prompt and input document. Here's how it is defined:

```python
def chatgpt_generation(prompt, document, model="gpt-3.5-turbo-0125"):
    """
    Generate an output based on a prompt and an input document.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.replace("[DOCUMENT]", document)}
    ]
    
    chat_completion = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=0
    )
    
    return chat_completion.choices[0].message.content
```
x??

---

#### Text Classification with Generative Models
Background context: The text describes using a generative model to classify documents as positive or negative movie reviews. It involves creating a template prompt and running the classification over a dataset.
:p What is the purpose of the `prompt` variable in the provided code?
??x
The `prompt` variable defines the instruction for generating a prediction on whether a document is a positive or negative movie review. Here’s the definition:

```python
prompt = """Predict whether the following document is a positive or negative  movie review: [DOCUMENT] If it is positive return 1 and if it is negative return 0. Do not give any other answers."""
```
x??

---

#### Implementing Exponential Backoff
Background context: The text mentions the importance of handling rate limit errors when using external APIs like OpenAI’s. It suggests implementing exponential backoff to prevent these errors.
:p What is the purpose of implementing exponential backoff in API requests?
??x
The purpose of implementing exponential backoff is to handle rate limit errors by temporarily pausing and retrying the request after a delay that increases with each failed attempt until the request succeeds or reaches a maximum number of retries. This helps avoid hitting the rate limits too frequently.
x??

---

