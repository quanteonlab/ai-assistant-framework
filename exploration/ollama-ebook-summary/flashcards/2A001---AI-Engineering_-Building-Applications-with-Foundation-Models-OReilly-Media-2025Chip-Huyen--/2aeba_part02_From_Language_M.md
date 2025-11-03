# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 2)

**Starting Chapter:** From Language Models to Large Language Models

---

---
#### Overview of Foundation Models
Foundation models are at the heart of AI's recent advancements. They represent large-scale, readily available models that bring both new opportunities and challenges for AI engineering. This evolution is crucial as it transforms how we approach building AI applications.
:p What are foundation models?
??x
Foundation models refer to large-scale machine learning models that have become a key catalyst in the rapid development of artificial intelligence. These models are characterized by their extensive size, which allows them to process vast amounts of data and perform complex tasks efficiently. They provide a robust foundation for developing various AI applications.
x??

---
#### From Language Models to Large Language Models
Language models have evolved from simple statistical representations of language to more complex self-supervised learning techniques that enable the growth to today's scale. The transition involves significant advancements in technology, leading to more sophisticated and versatile AI systems.
:p What is a language model?
??x
A language model encodes statistical information about one or more languages, providing insights into how likely certain words are to appear in specific contexts. For example, given the context "My favorite color is __," a language model that understands English would predict "blue" more often than "car."
x??

---
#### Self-Supervision
Self-supervised learning plays a crucial role in enabling large-scale language models by allowing them to learn from vast amounts of unlabeled data. This method differs from traditional supervised learning, which requires labeled datasets.
:p What is self-supervision?
??x
Self-supervision refers to the process where a model learns from unlabelled data by using some form of auxiliary task that helps it understand the context or structure of the input data. For example, in language models, predicting words based on their surrounding context without explicit labels is a common self-supervised learning technique.
x??

---
#### Statistical Nature of Languages
The statistical nature of languages was recognized centuries ago and has been pivotal in developing techniques like simple frequency analysis to decode messages. Claude Shannon's work in the 1950s further advanced this understanding with more sophisticated statistical methods.
:p How did Sherlock Holmes use statistical information?
??x
Sherlock Holmes used the statistical fact that the letter 'E' is the most common in English to deduce that the most frequent stick figure in a sequence of mysterious figures represented 'E'. This simple frequency analysis helped him decode the messages, showcasing how statistical insights can be applied to cryptographic problems.
x??

---
#### Entropy in Language Modeling
Entropy, introduced by Claude Shannon in his 1951 paper, is a measure used to model and understand the unpredictability or information content of languages. It quantifies the uncertainty in a set of possible outcomes.
:p What is entropy?
??x
Entropy is a measure of uncertainty or randomness in a system. In language modeling, it quantifies how unpredictable the next word might be given the current context. The concept was introduced by Claude Shannon and remains fundamental in understanding and predicting sequences of words in natural languages.
x??

---
#### Multilingual Language Models
Modern language models can handle multiple languages, leveraging statistical properties shared across different linguistic systems to improve overall performance and versatility.
:p How do modern language models differ from early ones?
??x
Modern language models are designed to handle multiple languages by incorporating the common statistical patterns found in various languages. This approach allows them to be more versatile and perform better when dealing with multilingual data, as they can draw on shared features across different linguistic systems.
x??

---

#### Tokenization Process
Tokenization is the process of breaking down the original text into smaller units called tokens. Depending on the model, a token can be a character, a word, or a part of a word (like -tion). This process allows models to understand and process language more efficiently.
:p What is tokenization?
??x
Tokenization is the process of dividing text into meaningful components called tokens, which can be characters, words, or parts of words. This helps models like GPT-4 break down text for efficient processing and understanding.
x??

---
#### Types of Language Models
Language models are categorized based on their ability to use context from different positions in a sequence of tokens. There are two main types: masked language models and autoregressive language models.
:p What are the two main types of language models?
??x
The two main types of language models are masked language models and autoregressive (or causal) language models. Masked language models predict missing tokens anywhere in a sequence using both preceding and following context, while autoregressive models generate text token by token, only considering past tokens.
x??

---
#### Autoregressive Language Models
Autoregressive language models, also known as causal language models, generate text token by token from left to right. They cannot use future context when predicting the next token in a sequence.
:p How do autoregressive language models work?
??x
Autoregressive language models generate text one token at a time, moving from left to right. At each step, they can only consider tokens that have already been generated and not any information from upcoming tokens. This makes them efficient for generating text but limits their ability to use future context.
x??

---
#### Tokenization in GPT-4
In the example provided, "I can’t wait to build AI applications" is broken into nine tokens: I,  can, 't,  wait,  to,  build,  AI,  application, s. The word “can't” is split into two tokens because it's a common contraction.
:p How does GPT-4 tokenize the phrase "I can’t wait to build AI applications"?
??x
GPT-4 tokenizes the phrase by splitting words and contractions appropriately. For example, "I can’t wait to build AI applications" is broken into nine tokens: I,  can, 't,  wait,  to,  build,  AI,  application, s. The word “can't” is split into two tokens because it's a common contraction.
x??

---
#### Vocabulary Size in Language Models
The vocabulary size of a model refers to the number of distinct tokens it can work with. For instance, GPT-4 has a vocabulary size of 100,256, while Mixtral 8x7B has a smaller vocabulary size of 32,000.
:p What is the importance of vocabulary size in language models?
??x
The vocabulary size is crucial because it defines how many distinct tokens (words or parts of words) a model can recognize and use. A larger vocabulary allows for more flexibility and accuracy in text generation and understanding but also increases computational requirements. For example, GPT-4 has a larger vocabulary than Mixtral 8x7B.
x??

---
#### Advantages of Using Tokens
Using tokens instead of characters or whole words offers several advantages:
1. **Meaningful Components**: Tokens allow breaking down words into meaningful components (e.g., "cooking" to "cook" and "ing").
2. **Efficiency**: Fewer unique tokens than unique words reduce the model's vocabulary size, making it more efficient.
3. **Handling Unknown Words**: Tokens help in understanding unknown or made-up words by splitting them into recognizable parts.

:p Why do language models prefer using tokens over characters or whole words?
??x
Language models prefer using tokens because:
1. Tokens allow breaking down words into meaningful components, providing deeper insights.
2. Fewer unique tokens than unique words reduce the model's vocabulary size, making it more efficient.
3. Tokens help in understanding unknown or made-up words by splitting them into recognizable parts.

This approach balances the need for fewer units while retaining more meaning than individual characters.
x??

---

#### Autoregressive Language Model
Background context: An autoregressive language model is a type of model that predicts the next token in a sequence based only on the preceding tokens. This makes it capable of generating one token after another, creating an open-ended output that can be used for various text generation tasks.
:p What is an autoregressive language model?
??x
An autoregressive language model is trained to predict the next token in a sequence using only the preceding tokens. It generates text by predicting one token at a time and uses the generated sequence as input for the next prediction, creating an open-ended output that can be used for various tasks such as text completion.
x??

---

#### Text Generation with Autoregressive Language Models
Background context: Autoregressive language models are currently the preferred choice for text generation due to their capability of continually generating one token after another. They are more popular than masked language models, which can also generate text but require more effort.
:p Why are autoregressive language models preferred over masked language models for text generation?
??x
Autoregressive language models are preferred because they can continuously generate text by predicting the next token based on the current sequence of tokens. They do not require additional steps to mask or unmask tokens, making them simpler and more efficient for generating text.
x??

---

#### Generative AI and Language Models
Background context: A generative model like a language model produces open-ended outputs using its fixed vocabulary to construct an infinite number of possible sequences. This makes the output unpredictable and varied, which is why it's called "generative."
:p How do generative models differ from other types of models in terms of output?
??x
Generative models, including language models, produce open-ended outputs by constructing a sequence using their fixed vocabulary. Unlike deterministic or rule-based systems, these models generate text that can vary widely, making the output unpredictable and diverse.
x??

---

#### Language Model as a Completion Machine
Background context: A language model can be thought of as a completion machine where it takes a given prompt and tries to complete it by predicting subsequent tokens. This is useful in various applications like translation, summarization, coding, and solving math problems.
:p How does a language model work when presented with a prompt?
??x
A language model works by taking a prompt (text) as input and predicting the next token(s) that would likely follow based on its training data. This process can be iterated to generate a sequence of text that completes or expands upon the given prompt.
x??

---

#### Example of Language Model Completion
Background context: The example provided demonstrates how a language model completes a sentence by adding tokens that fit the context, such as “that is the question” in response to "To be or not to be".
:p Can you provide an example of a completion task performed by a language model?
??x
Sure, given the prompt "To be or not to be," a language model might complete it with "that is the question." This shows how the model predicts the next token(s) that fit the context.
x??

---

#### Excitement and Frustration of Language Models
Background context: The probabilistic nature of language models makes them both exciting and frustrating. They generate text based on probabilities, which means predictions are not guaranteed to be correct, leading to creative but sometimes inaccurate outputs.
:p Why do language models create excitement and frustration?
??x
Language models create excitement due to their ability to produce creative and diverse outputs based on probabilities. However, they can also frustrate users because the generated text is not always accurate or as intended, making it challenging to rely solely on these models for precise information.
x??

---

#### Completion Tasks Versus Conversations
Background context: While completion tasks are powerful and useful in many applications, they do not fully replicate human-like conversational engagement. A model might respond with another question instead of directly answering a user's query.
:p How does the concept of completion differ from engaging in a conversation?
??x
Completion involves generating text to extend or complete an existing prompt, whereas true conversation requires understanding and responding appropriately to specific queries. While completion can generate useful responses, it may not always address the user's intent fully, unlike human-like conversations which involve back-and-forth exchanges.
x??

---

#### Self-Supervised Learning in Language Models
Background context: Self-supervision is a method that allows language models to be trained without explicit labeled data. It helps overcome the bottleneck of obtaining large datasets with labeled data, making it easier to scale up model training.
:p What is self-supervision and why is it important for language models?
??x
Self-supervision is a training approach where language models learn from unlabeled data by predicting certain parts of the input sequence. This method helps overcome the need for expensive and slow processes of obtaining labeled data, allowing larger datasets to be used for model training.
x??

---

#### The Scaling Approach in AI Engineering
Background context: The scaling approach that led to the "ChatGPT moment" involved using self-supervised learning instead of traditional supervised learning. This made it possible to train larger models with more data, leading to significant advancements and breakthroughs in AI applications.
:p How did self-supervision contribute to the recent advances in language models?
??x
Self-supervision contributed by enabling the training of larger models on vast amounts of unlabeled data. This approach bypassed the need for expensive labeled datasets, allowing models like ChatGPT to scale up and achieve remarkable performance in various tasks.
x??

---

---
#### Supervised Learning Overview
Background context explaining supervised learning, where models are trained on labeled data to predict outcomes. The model learns from examples that have explicit labels indicating correct behavior or output.

:p What is supervised learning?
??x
Supervised learning involves training a model using datasets that include both input features and corresponding output labels. The goal is for the model to learn a mapping function from inputs to outputs so it can accurately predict outcomes for new, unseen data.
x??

---
#### Data Labeling Costs
Explanation of how the cost of labeling data varies based on factors like task complexity, scale, and provider.

:p How do costs associated with data labeling vary?
??x
Data labeling costs can vary significantly depending on several factors. The complexity of the task, the size of the dataset, and the provider's pricing model all influence the overall cost. For instance, Amazon SageMaker Ground Truth charges 8 cents per image for fewer than 50,000 images but reduces to 2 cents per image for more than 1 million images.

```java
public class LabelingCosts {
    public static double calculateCost(int numImages) {
        if (numImages < 50000) {
            return numImages * 0.08; // 8 cents
        } else if (numImages > 1000000) {
            return numImages * 0.02; // 2 cents
        } else {
            return -1; // Error case
        }
    }
}
```
x??

---
#### AlexNet and Its Impact on Deep Learning
Explanation of the impact of AlexNet, which started the deep learning revolution in the 2010s by classifying images into over 1,000 categories.

:p What was significant about AlexNet?
??x
AlexNet marked a significant milestone in the deep learning era. It was trained to classify more than 1 million images from the ImageNet dataset into one of 1,000 categories, including objects like "car," "balloon," or "monkey." This success demonstrated the potential of deep convolutional neural networks for image recognition tasks.
x??

---
#### Self-Supervised Learning
Explanation of self-supervised learning where models infer labels from input data without explicit labels.

:p What is self-supervised learning?
??x
Self-supervised learning involves training models to predict or generate information that can be inferred from the input data itself. This method eliminates the need for explicit labeling, making it more cost-effective and scalable. An example of self-supervised learning is language modeling, where each sentence provides both context and target labels.
x??

---
#### Language Modeling Examples
Explanation using the provided example to illustrate how a single sentence can generate multiple training samples.

:p How does language modeling work with sentences?
??x
Language modeling generates training samples from input sequences by predicting tokens in a sequence. For instance, the sentence "I love street food." provides six possible training samples:
```plaintext
<BOS> I <BOS>, I love <BOS>, I, love street <BOS>, I, love, street food <BOS>, I, love, street, food . <BOS>, I, love, street, food, . <EOS>
```
Here, `<BOS>` marks the beginning of a sequence and `<EOS>` indicates its end. The model learns to predict the next token in each context.
x??

---

#### Model Parameters and Their Significance
Background context: In machine learning, model parameters are variables within a model that are updated during training to optimize its performance. These include both weights (model parameters) and biases. Historically, the terms have been used separately, but today, "weights" is commonly used to refer to all parameters.
:p What do we generally use to refer to model parameters in modern machine learning?
??x
In modern machine learning, we typically use "model weights" to refer to all parameters, which include both weights and biases. This terminology simplifies the language but can be confusing if not aware of historical context.
x??

---

#### Why Larger Models Require More Data
Background context: Larger models have greater capacity for learning complex behaviors, hence they require more data during training to maximize performance. However, smaller models might suffice with less data due to their lower complexity and computational requirements.
:p Why do larger models need more data compared to smaller models?
??x
Larger models have a higher capacity to learn intricate patterns in the data, requiring more training data to ensure that they generalize well. Smaller models, being simpler, may require less data to achieve similar performance due to their reduced complexity and computational demands.
x??

---

#### Self-Supervision vs. Unsupervised Learning
Background context: Self-supervised learning involves generating labels from the input data itself, while unsupervised learning does not use any labels at all. This allows models like language models to learn from vast amounts of unlabeled text data without manual labeling.
:p What distinguishes self-supervised learning from unsupervised learning?
??x
Self-supervised learning generates labels from the input data, whereas unsupervised learning doesn't require any labeled data. Self-supervised learning is useful for processing large volumes of text where manual labeling would be impractical.
x??

---

#### Foundation Models and Their Evolution
Background context: Foundation models are large-scale language or multimodal models that can perform multiple tasks due to their extensive training on diverse datasets. They represent a shift from task-specific models to general-purpose ones capable of handling various applications.
:p What is the significance of foundation models in AI?
??x
Foundation models, like large language models and large multimodal models, are significant because they offer broad applicability across different tasks without needing separate specialized models for each. Their extensive training on diverse data allows them to handle a wide range of applications effectively.
x??

---

#### The Need for Data with Larger Models
Background context: With the increase in model size comes greater capacity to learn complex behaviors, necessitating more training data to ensure optimal performance and generalization. However, smaller models might suffice with less data due to their lower complexity and computational requirements.
:p Why is larger model size associated with needing more training data?
??x
Larger model sizes come with increased capacity to capture intricate patterns in the data. To fully utilize this capacity and avoid overfitting, more training data is needed to ensure that the model generalizes well across different scenarios. Smaller models require less data due to their simpler architecture.
x??

---

#### Multimodal Models and Their Applications
Background context: Multimodal models can process various types of data such as text, images, or videos, making them more versatile than single-modal (text-only) models. These models are crucial for applications that require understanding of multiple sensory inputs.
:p What is the difference between a language model and a multimodal model?
??x
A language model processes only textual data, while a multimodal model can handle various types of data such as text, images, or videos. This makes multimodal models more versatile and capable of handling complex tasks requiring understanding from multiple sensory inputs.
x??

---

#### The Role of Embeddings in Foundation Models
Background context: Embedding models like CLIP produce joint embeddings for texts and images, enabling better understanding and processing of diverse data types. These embeddings capture the meanings of original data points effectively.
:p What is an embedding model?
??x
An embedding model produces vectors that represent the meanings of original data points, such as text or images. Embedding models like CLIP create joint embeddings for texts and images, allowing better understanding and processing of diverse data types.
x??

---

#### Transition from Task-Specific to General-Purpose Models
Background context: Traditional AI models were often designed for specific tasks, limiting their applicability. Foundation models, being larger and more general, can perform multiple tasks with relative ease after some fine-tuning or additional training.
:p How do foundation models represent a shift in AI development?
??x
Foundation models mark a transition from task-specific models to general-purpose ones capable of handling various applications. They are trained on extensive data, enabling them to perform multiple tasks effectively and offering more flexibility than specialized models.
x??

---

#### Techniques for Fine-Tuning Models
Background context: Prompt engineering, retrieval-augmented generation (RAG), and fine-tuning are techniques used to adapt pre-trained foundation models to specific applications. These methods help tailor the model's performance according to user needs without extensive retraining.
:p What are some common AI engineering techniques for adapting a model?
??x
Common AI engineering techniques include prompt engineering, retrieval-augmented generation (RAG), and fine-tuning. These methods allow you to adapt pre-trained models to specific tasks more effectively by providing detailed instructions, leveraging databases, or further training on relevant data.
x??

---

#### Adapting Models Versus Building from Scratch

Adaptation of existing models is generally easier, quicker, and less resource-intensive compared to building a model from scratch. For example, adapting might require only 10 examples over a weekend, whereas building from scratch could need millions of examples and six months. 

This approach uses foundation models, which are powerful due to their ability to perform more tasks beyond existing ones, making AI applications more accessible.

:p How does the adaptation of an existing model compare to building one from scratch in terms of effort and resources?
??x
Adapting an existing model typically requires fewer examples (e.g., 10) and less time (e.g., a weekend), whereas building a model from scratch might require millions of examples and six months. This makes adaptation cheaper and quicker.
x??

---

#### Benefits of Task-Specific Models

Task-specific models can be smaller, making them faster and cheaper to use than generic foundation models.

:p What are the benefits of using task-specific models over foundation models?
??x
Task-specific models can be more efficient in terms of size, which translates into faster performance and lower costs. They offer a balance between customization and resource efficiency.
x??

---

#### AI Engineering as an Discipline

AI engineering refers to building applications on top of foundation models. It has emerged due to the availability of powerful foundation models.

:p What is AI engineering and why is it significant?
??x
AI engineering involves building applications using existing, powerful foundation models rather than developing new models from scratch. Its significance lies in leveraging pre-existing models to accelerate application development.
x??

---

#### Factors Driving AI Engineering

Three factors are driving the growth of AI engineering: general-purpose AI capabilities, increased investments, and low entrance barriers.

:p What are the three factors contributing to the rapid growth of AI engineering?
??x
The three factors are:
1. General-purpose AI capabilities enabling more applications.
2. Increased AI investments due to successes like ChatGPT.
3. Low entrance barriers through model-as-a-service approaches.
x??

---

#### Example Applications of AI

AI can be used in various tasks, including writing emails, responding to customer requests, and creating marketing materials.

:p What are some examples of how AI is currently being utilized?
??x
Examples include:
- Writing emails
- Responding to customer requests
- Explanations of complex contracts
- Generating images and videos for marketing purposes
- Automating tasks that require communication.
x??

---

#### Investment in AI

Investment in AI has seen a significant increase, with companies incorporating it into their products and processes.

:p What trend is observed in AI investment from 2018 to 2023?
??x
The number of S&P 500 companies mentioning AI in earnings calls increased significantly. For instance, one in three companies mentioned AI in the second quarter of 2023, compared to just a third the previous year.
x??

---

#### Rise of AI Engineering Tools

AI engineering tools are gaining traction rapidly due to their ease of use and low barrier to entry.

:p What trend is observed with open-source AI engineering tools?
??x
Open-source AI engineering tools have gained significant traction. For example, AutoGPT, Stable Diffusion UI, LangChain, and Ollama have more GitHub stars than Bitcoin in just two years.
x??

---

#### Terminology for AI Engineering

AI engineering is the preferred term over ML engineering or MLOps because it better captures the process of adapting existing models.

:p Why did the author choose "AI engineering" as the term for this book?
??x
The author chose "AI engineering" because it best describes the process of building applications using foundation models, differentiating it from traditional ML engineering. This term was also supported by a survey indicating that most people prefer it.
x??

---

