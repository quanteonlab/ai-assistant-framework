# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 25)

**Starting Chapter:** Finetuning Overview

---

#### Finetuning Overview
Finetuning is a method of adapting a model to a specific task by further training it, either on new data or with modified parameters. It aims to enhance the model’s performance for particular tasks, such as coding, medical question answering, or adhering to specific output styles and formats.
:p What is finetuning?
??x
Finetuning involves taking a pre-trained model and retraining its weights on task-specific data to improve its performance on that task. It differs from prompt-based methods, which adapt models by providing them with instructions and context without altering the underlying architecture.
x??

---

#### Memory Footprint of Finetuning
Compared to prompt-based methods, finetuning requires a much higher memory footprint due to the need for training more parameters. For large foundation models, naive finetuning often necessitates more memory than can fit on a single GPU.
:p Why is finetuning memory-intensive?
??x
Finetuning is memory-intensive because it involves adjusting all or part of the pre-trained model's weights using new data. This process requires significant computational resources and can exceed the capacity of a single GPU, making it expensive and challenging to execute without specialized hardware or distributed computing setups.
x??

---

#### PEFT (Parameter-Efficient Finetuning)
PEFT is a memory-efficient approach that has become dominant in finetuning space. It aims to reduce the number of parameters that need to be fine-tuned, thus lowering memory requirements while still improving model performance.
:p What is PEFT?
??x
PEFT, or Parameter-Efficient Finetuning, is an approach designed to reduce the computational resources required for finetuning large models by efficiently adjusting a subset of the model's parameters. It aims to balance between full fine-tuning and prompt-based methods, offering a middle ground with better performance than naive finetuning.
x??

---

#### Adapter-Based Techniques
Adapter-based techniques are part of PEFT and involve adding small, task-specific modules (adapters) to pre-trained models. These adapters can be fine-tuned independently of the rest of the model, making them more memory-efficient and easier to manage.
:p What are adapter-based techniques?
??x
Adapter-based techniques involve incorporating small, specialized components called "adapters" into a pre-trained model. These adapters can be fine-tuned separately from the main model parameters, allowing for efficient finetuning with reduced memory footprint. This makes it possible to adapt large models to specific tasks without fully retraining them.
x??

---

#### Choosing Between Finetuning and Other Methods
When deciding between finetuning and other methods like RAG (Retrieval-Augmented Generation), consider factors such as the model's domain-specific capabilities, safety needs, and instruction-following requirements. The choice depends on whether you need more specialized knowledge or a broader, adaptable approach.
:p How do you decide when to use finetuning versus other methods?
??x
When deciding between finetuning and alternative methods like RAG (Retrieval-Augmented Generation), consider the following:
- **Domain-Specific Capabilities**: If your task requires deep expertise in specific domains like coding or medical questions, finetuning can enhance these capabilities.
- **Safety Needs**: To ensure model behavior aligns with safety protocols, finetuning might be necessary to adjust output styles and formats.
- **Instruction-Following Ability**: Finetuning is particularly useful for improving a model's ability to follow instructions precisely.

If you need more adaptability or less specialized knowledge, other methods like RAG might suffice. The choice depends on your specific requirements and constraints.
x??

---

#### ML Basics and Knowledge Required
While the basics of machine learning are beyond the scope of this book, having some understanding can be helpful when dealing with finetuning. Key concepts include training data, model architecture, optimization algorithms, and loss functions.
:p Why is knowledge about ML necessary for finetuning?
??x
Knowledge about machine learning is necessary for finetuning because it involves adjusting a pre-trained model's parameters based on new data. Understanding concepts like training data, model architecture, optimization algorithms (e.g., Adam or SGD), and loss functions helps in effectively fine-tuning the model. Without this knowledge, you might struggle to make informed decisions about which parts of the model to adjust and how.
x??

---

#### Transfer Learning
Transfer learning is a broader concept that focuses on transferring knowledge from one task to another related task. Finetuning can be seen as an application of transfer learning, where pre-trained models are adapted for new tasks by adjusting their parameters based on specific data.
:p What is the relationship between finetuning and transfer learning?
??x
Finetuning is a form of transfer learning. It involves using a model that has been trained on one task to perform well on another related task by fine-tuning its parameters with new, relevant data. This approach leverages pre-existing knowledge in the model to accelerate learning for the new task.
x??

---

#### Transfer Learning Overview
Transfer learning allows a model trained on one task to be adapted for another related task with fewer resources. This is particularly valuable when there's limited or expensive training data available for the target task.

:p What is transfer learning, and why is it beneficial?
??x
Transfer learning is a technique where knowledge gained from solving one problem can be applied to another, related but different, problem. It helps in situations with limited data by leveraging pre-trained models that have already learned general features or patterns which can then be adapted for the specific task at hand.

For example, if you train a model on a large dataset like ImageNet and later want it to recognize medical images, transfer learning allows you to use the pre-trained model’s knowledge of recognizing visual features but adjust its final layers to classify medical conditions.
x??

---

#### Sample Efficiency in Transfer Learning
Transfer learning improves sample efficiency by allowing models to learn the same behavior with fewer examples compared to training from scratch.

:p How does transfer learning improve sample efficiency?
??x
Sample efficiency refers to how well a model can generalize after being trained on a relatively small amount of data. By leveraging pre-trained models, you can fine-tune a model to perform specific tasks using much less labeled data than would be required if starting from scratch. This is because the pre-trained model already has learned some basic features and patterns.

For instance, in legal question answering, while training a model from scratch might require millions of examples, fine-tuning an existing model trained on text completion might only need a few hundred.
x??

---

#### Feature-Based Transfer Learning
Feature-based transfer learning involves using the features or embeddings extracted by one model as input for another.

:p What is feature-based transfer learning?
??x
Feature-based transfer learning is a method where a pre-trained model is used to extract features from data, which are then passed on to another model. This approach is common in computer vision tasks, such as using a pre-trained model like VGG or ResNet trained on ImageNet and applying its output embeddings (features) to new tasks like object detection.

For example, you might train a model on the ImageNet dataset for image classification and then use this model’s feature extraction layers to create features from images which are further used in more specific tasks such as segmentation.
x??

---

#### Finetuning vs. Pre-Training
Finetuning is an extension of pre-training where a model is adapted for a specific task after being pretrained on general data.

:p How does finetuning differ from pre-training?
??x
Pre-training involves training a model on large amounts of unlabeled or weakly labeled data using self-supervised methods to learn basic patterns and features. Finetuning, on the other hand, is the process of adapting this pre-trained model for specific tasks by fine-tuning its parameters with task-specific data.

For example, you might start by pre-training a language model on a large text corpus like Wikipedia or BooksCorpus, then use supervised finetuning to adapt it for legal question answering. This way, the model can leverage its learned knowledge and just refine its behavior based on the specific task.
x??

---

#### Supervised Finetuning
Supervised finetuning involves training a pre-trained model using labeled data for a specific task.

:p What is supervised finetuning?
??x
Supervised finetuning is a method where a pre-trained model is further trained with labeled data to adapt it to a specific task. This process typically involves adding a new layer or layers (such as a classifier head) and training the entire model end-to-end on the task-specific dataset.

For example, if you have a pre-trained language model that has learned general text understanding from a large corpus of text, you can fine-tune it by adding a few additional layers to classify legal questions into their respective answers. The model will then learn how to map inputs (legal questions) to outputs (answers) using the provided labeled data.
x??

---

#### Preference Finetuning
Preference finetuning involves training models on tasks where labels are implicit or involve ranking and preferences.

:p What is preference finetuning?
??x
Preference finetuning is a form of finetuning where the model is trained to learn from implicitly labeled data, often through pairwise comparisons. This approach is useful when the task requires understanding the relative preferences between different items rather than absolute labels.

For example, you might train a recommendation system by comparing pairs of items and indicating which one is preferred. The model learns the underlying preferences based on these implicit feedback signals.
x??

---

#### Finetuning Process
Finetuning involves fine-tuning a pre-trained model’s parameters with task-specific data to adapt it for a specific application.

:p What does the finetuning process involve?
??x
The finetuning process starts by taking a pre-trained model and adjusting its parameters using labeled data specific to the target task. This can be done in several ways, including supervised finetuning where the entire model is trained end-to-end on the new dataset or preference finetuning where the model learns from implicit preferences.

For example, if you have a text classifier that was initially pre-trained on a large corpus of general text, you might fine-tune it by adding more layers and training it with annotated legal question-answer pairs. The process involves retraining parts of the model while keeping some layers fixed or freezing them.
x??

---

---
#### Autoregressive vs Masked Models
Autoregressive models predict the next token in a sequence based on previous tokens, while masked models use both preceding and following tokens to fill in the blank.

:p What is the difference between autoregressive and masked language models?
??x
In an autoregressive model, the task is to predict the next token given all previous tokens. For example, if we have a sequence "I am going to", the model predicts "go" based on "I am going". In contrast, a masked model predicts missing tokens in a sequence like "I [MASK] to go", where it fills in "[MASK]" with "am".

Code Example (Pseudocode):
```python
def autoregressive_prediction(sequence):
    # Predict next token given previous tokens
    pass

def masked_filling(sequence):
    # Fill in the blank using both preceding and following context
    pass
```
x??

---
#### Supervised vs Self-Supervised Finetuning
Supervised finetuning uses labeled data to refine a model, aligning it with human preferences. In contrast, self-supervised finetuning (also called continued pre-training) relies on unlabeled data for initial learning.

:p What are the differences between supervised and self-supervised finetuning?
??x
In supervised finetuning, you train a model using labeled data where each input is paired with an output. For instance, in text generation tasks, inputs might be instructions, and outputs could be responses. This type of finetuning helps align the model more closely with human preferences.

Self-supervised finetuning, on the other hand, involves continuing pre-training by training a model without explicit labels but leveraging large amounts of data. The model learns to generate coherent text or complete sequences based on partial inputs. For example, it might predict missing words in a sentence.

Code Example (Pseudocode):
```python
# Supervised Finetuning
def supervised_finetuning(inputs, outputs):
    # Train the model using paired input-output data
    pass

# Self-Supervised Finetuning
def self_supervised_finetuning(input_sequences):
    # Continue pre-training with unlabeled data
    pass
```
x??

---
#### Infilling Finetuning
Infilling finetuning involves filling in the blank using both preceding and following tokens. This technique is particularly useful for tasks like text editing or code debugging.

:p What is infilling finetuning, and why is it important?
??x
Infilling finetuning focuses on predicting a missing token within a sentence or sequence based on surrounding context. For example, given the input "I went to [MASK] with my friends", an infilling model would predict "[MASK]" as "the park".

This technique is crucial for tasks such as text editing where you might want to automatically complete parts of a document.

Code Example (Pseudocode):
```python
def infilling_finetuning(input_sequences, mask_positions, targets):
    # Train the model to fill in missing tokens based on context
    pass
```
x??

---
#### Long-Context Finetuning
Long-context finetuning extends the model's ability to process longer sequences by modifying its architecture, such as adjusting positional embeddings.

:p How does long-context finetuning differ from other types of finetuning?
??x
Long-context finetuning is a specialized form of finetuning designed to handle very long input sequences. Unlike standard finetuning methods that might struggle with long inputs due to limitations in memory or computational resources, long-context finetuning involves modifying the model’s architecture (e.g., adjusting positional embeddings) to support longer sequences.

For instance, if a model is trained on 4,096 tokens but needs to process up to 16,384 tokens for a specific task, long-context finetuning can help by ensuring that the positional encoding scheme can handle the increased length effectively without degrading performance on shorter inputs.

Code Example (Pseudocode):
```python
def modify_positional_embeddings(model):
    # Adjust positional embeddings to support longer sequences
    pass

# Example of extending context length
modified_model = modify_positional_embeddings(original_model)
```
x??

---
#### Instruction Finetuning
Instruction finetuning involves training a model with (input, output) pairs where inputs are instructions and outputs are responses. This is also referred to as supervised finetuning.

:p What is instruction finetuning, and how does it differ from other types of finetuning?
??x
Instruction finetuning specifically refers to the process of fine-tuning a model using (input, output) pairs where inputs are instructions or prompts, and outputs are expected responses. This form of supervised finetuning aligns the model's behavior more closely with human preferences and is particularly useful for tasks that require precise response generation based on given instructions.

For example, if you want to train a model to generate summaries from book chapters, you would provide it with inputs like "Please summarize Chapter 1" and outputs like "Chapter 1 was about the introduction of the main characters and setting." This method ensures the model can produce appropriate responses according to specific guidelines or tasks.

Code Example (Pseudocode):
```python
def instruction_finetuning(instructions, responses):
    # Train the model using instructions and corresponding responses
    pass
```
x??

---
#### Preference Finetuning
Preference finetuning uses comparative data in the format of (instruction, winning response, losing response) to train a model to generate preferred outputs.

:p What is preference finetuning, and how does it work?
??x
Preference finetuning involves training a model using triplets of data where each triplet consists of an instruction, a "winning" response that aligns with human preferences, and a "losing" response that doesn't. The goal is to teach the model which responses are more desirable based on these examples.

This approach ensures that the model's outputs better match human preferences by learning from explicit comparisons between good and bad responses.

Code Example (Pseudocode):
```python
def preference_finetuning(instructions, winning_responses, losing_responses):
    # Train the model to favor winning responses over losing ones
    pass
```
x??

---

---

#### When to Finetune
Background context explaining when finetuning is appropriate, noting that it typically comes after extensive experimentation with prompt-based methods. Discuss the resources required for finetuning versus prompting.

:p When should you consider finetuning a model?
??x
You should consider finetuning a model after extensive experiments with prompt-based methods have been conducted. Finetuning requires significantly more resources, including data and hardware, as well as ML talent, compared to simpler prompt-based methods. Therefore, it is generally attempted only when other options have not sufficiently met your needs.
x??

---

#### Reasons to Finetune
Background context explaining the primary reasons for finetuning a model, such as improving general capabilities or task-specific abilities.

:p Why should you choose to fine-tune a pre-trained model?
??x
You should choose to fine-tune a pre-trained model if it has not been sufficiently trained on your specific task. Finetuning can help improve the model's ability to generate outputs following specific structures, such as JSON or YAML formats. It is particularly useful when an out-of-the-box model performs well in general but struggles with niche tasks.
x??

---

#### Bias Mitigation Through Finetuning
Background context explaining how finetuning can be used to mitigate biases in models by exposing them to carefully curated data.

:p How can finetuning help mitigate bias in a model?
??x
Finetuning can help mitigate bias in a model by exposing it to carefully curated data during the fine-tuning process. For example, if a model consistently assigns male-sounding names to CEOs, finetuning it on a dataset with many female CEOs can counteract this bias. Gari mella et al. (2022) found that finetuning BERT-like language models on text authored by women can reduce gender biases, while finetuning them on texts by African authors can reduce racial biases.
x??

---

#### Finetuning Larger Models
Background context explaining the importance of distillation in fine-tuning smaller models to imitate larger ones.

:p Why is it common to use a small model for distillation?
??x
It is common to use a small model for distillation because smaller models require less memory and are cheaper and faster to use in production. A common approach is to finetune a small model to imitate the behavior of a larger model using data generated by this large model. This process, known as distillation, involves distilling the larger model's knowledge into the smaller model.
x??

---

#### Distillation Process
Background context explaining the concept and purpose of model distillation.

:p What is model distillation?
??x
Model distillation is a technique where a small model is finetuned to imitate the behavior of a larger, more complex model. The process involves using data generated by the large model to train the smaller model, effectively distilling the knowledge from the larger model into the smaller one. This approach helps in reducing memory requirements and improving efficiency.
x??

---

#### Finetuning Models on Specific Tasks
Finetuning is a process where an existing model, often pretrained on large datasets (like natural language understanding), is adapted to perform better on specific tasks. This can lead to improved performance compared to using out-of-the-box models but requires careful consideration of the trade-offs involved.

For instance, Grammarly found that their finetuned Flan-T5 models outperformed a GPT-3 variant specialized in text editing despite being significantly smaller (60 times). The finetuning process used only 82,000 pairs of instructions and outputs, which is much less than typically needed to train such a model from scratch.

:p What are the benefits of finetuning models on specific tasks?
??x
Finetuning can lead to better performance on targeted tasks compared to using out-of-the-box large pretrained models. This is because fine-tuning allows the model to learn task-specific nuances and generalizations, potentially reducing the need for large amounts of training data.

For example, a finetuned Flan-T5 model for text editing might perform as well or better than a much larger GPT-3 variant specialized in this domain but requires significantly less computational resources. This can be advantageous when dealing with limited computing power or time constraints.
x??

---

#### Comparison Between Finetuning and General Models
Finetuning involves adapting a pretrained model to a specific task by training it further on relevant data, whereas using out-of-the-box models relies on the general knowledge learned during initial training.

A key difference is that while finetuning can improve performance for the targeted tasks, it may degrade performance on other unrelated tasks. This phenomenon highlights the importance of balancing specialization and versatility in model design and deployment.

:p How does finetuning affect a model’s performance across different tasks?
??x
Finetuning typically enhances performance on the specific task or set of tasks used during fine-tuning but can sometimes result in reduced performance on other unrelated tasks. This is because the model becomes too specialized, losing its ability to generalize well beyond the trained tasks.

For example, if a text editor model is finetuned for changing orders, it might perform poorly when asked about general feedback or product recommendations, even though these were not part of the training data.
x??

---

#### Data Requirements and Sources for Finetuning
The amount of annotated data required for effective finetuning can vary widely depending on the complexity of the task. While smaller models like Flan-T5 (used in Grammarly's example) can achieve competitive results with as few as 82,000 pairs, larger models may require more extensive datasets.

Open-source and AI-generated data can help reduce costs but their effectiveness is variable and must be carefully evaluated for each use case.

:p What are the challenges associated with obtaining annotated data for finetuning?
??x
Obtaining annotated data for finetuning poses several challenges. First, manually annotating large datasets can be time-consuming and expensive, especially when tasks require domain expertise or critical thinking. Second, even open-source and AI-generated data may not always meet the quality standards required for effective training.

For instance, while using a smaller model like Flan-T5 with 82,000 pairs of instructions and outputs might be sufficient, larger models often need much more data to maintain performance levels.
x??

---

#### Trade-offs in Finetuning
While finetuning can improve model performance on specific tasks, it comes with trade-offs. Carefully crafted prompts and context can also enhance a model's performance without the need for fine-tuning. Additionally, structured output techniques discussed earlier (in Chapter 2) can be effective alternatives.

However, finetuning a model specifically for one task may lead to reduced performance on other tasks. This is particularly problematic if your application requires handling multiple types of queries or inputs.

:p What are some reasons not to use finetuning?
??x
There are several reasons why finetuning might not be the best approach:

1. **Degraded Performance on Other Tasks**: Finetuning a model for one specific task can lead to reduced performance on other unrelated tasks.
2. **Complexity and Resource Requirements**: Finetuning requires substantial resources, including annotated data, which can be costly and time-consuming to acquire, especially when the domain demands specialized knowledge.
3. **Alternative Methods**: Carefully crafted prompts and context, or other structured output techniques discussed in Chapter 2, might achieve similar performance improvements without the need for fine-tuning.

For instance, if your application needs to handle multiple types of queries (like product recommendations, changing orders, and general feedback), finetuning on just one type of query could degrade performance on the others.
x??

---

#### Understanding Finetuning and Its Implications

Background context: Finetuning is a process used to adjust pre-trained models for specific tasks, often involving tweaking training parameters, monitoring learning processes, and evaluating model performance. This process requires understanding various technical aspects such as optimizers, learning rates, data requirements, and evaluation metrics.

:p What are the key steps involved in finetuning a pre-trained model?
??x
The key steps involved in finetuning a pre-trained model include:

1. **Understanding Optimizers**: Optimizers like Adam or SGD adjust weights during training to minimize loss functions.
2. **Choosing Learning Rate**: A learning rate determines how quickly the model learns from data; too high and it may overshoot, too low and it might take too long to converge.
3. **Data Collection**: Ensuring you have adequate labeled data for your specific task is crucial.
4. **Addressing Overfitting/Underfitting**: Techniques such as regularization or increasing training data can help balance model complexity.

Example of a simple Adam optimizer setup in Python:
```python
import torch

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```

x??

---

#### Model Serving Considerations

Background context: Once you have finetuned a model, the next step is deciding how to serve it. This involves choosing between hosting it yourself or using an API service. Serving models efficiently requires optimization for inference, especially for large language models (LLMs).

:p How do you decide whether to host your finetuned model in-house or use an API service?
??x
Deciding whether to host a finetuned model in-house or use an API service depends on several factors:

- **In-House Hosting**: If the company already has infrastructure for hosting models, it might be more convenient and cost-effective.
- **API Service**: Cloud services like AWS Lambda, Google Cloud Functions, or Azure Functions can provide scalable solutions with minimal setup and maintenance.

Example of setting up a simple API using Flask in Python:
```python
from flask import Flask, request

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict(data)
    return {'prediction': prediction}

if __name__ == '__main__':
    app.run()
```

x??

---

#### Monitoring and Maintaining Models

Background context: After deploying a finetuned model, ongoing monitoring and maintenance are necessary to ensure its performance. This includes setting up policies for model updates, comparing new base models, and deciding when to switch to a better-performing one.

:p What factors should be considered when deciding whether to switch to a new base model?
??x
Factors to consider when deciding whether to switch to a new base model include:

1. **Performance Improvement**: Measure the performance difference between your current finetuned model and the new base model on your specific task.
2. **Cost-Benefit Analysis**: Evaluate if the improvement justifies the effort of retraining or switching models, especially in terms of computational resources and time.
3. **Iterative Testing**: Continuously test both models to understand their performance under different conditions.

Example pseudo-code for comparing two models:
```python
def compare_models(model1, model2):
    # Define evaluation metrics
    metrics = evaluate_models_on_test_data([model1, model2])
    if max(metrics) > threshold:
        return True  # Switch to the better performing model
    else:
        return False
```

x??

---

#### The Role of Prompting in Finetuning

Background context: Prompting is a critical part of finetuning models, especially for tasks that require natural language understanding. It involves providing clear instructions and examples to guide the model's behavior.

:p Why might prompting be considered sufficient before resorting to finetuning?
??x
Prompting can often be sufficient before resorting to finetuning because:

- **Simplicity**: Prompt tuning requires fewer changes compared to full retraining.
- **Effectiveness**: Well-crafted prompts can significantly improve model behavior without the complexity and resource demands of finetuning.

Example of a well-crafted prompt:
```python
prompt = "Given the sentence 'The cat is on the mat,' please generate its past tense form."
```

x??

---

#### The Evolution of General-Purpose Models

Background context: General-purpose models, like those used in pre-trained language models (PLMs), are becoming increasingly capable and can often outperform domain-specific models due to their broader training.

:p Why might a general-purpose model perform better than a domain-specific one?
??x
A general-purpose model may perform better than a domain-specific one because:

- **Broader Training**: General-purpose models are trained on diverse datasets, allowing them to generalize well across various tasks and domains.
- **Reduced Customization**: Domain-specific models require custom training data and adjustments, which can be resource-intensive.

Example of a general-purpose model outperforming a domain-specific model:
```python
# Assume `general_model` is a pre-trained language model and `domain_specific_model` is trained on a specific dataset.
from sklearn.metrics import accuracy_score

general_predictions = general_model.predict(test_data)
specific_predictions = domain_specific_model.predict(test_data)

general_accuracy = accuracy_score(true_labels, general_predictions)
specific_accuracy = accuracy_score(true_labels, specific_predictions)

if general_accuracy > specific_accuracy:
    print("General-purpose model performs better.")
else:
    print("Domain-specific model outperforms general-purpose model.")
```

x??

---

#### BloombergGPT Overview
Bloomberg introduced BloombergGPT, a mid-size model (50 billion parameters) in March 2023. The primary goal was to develop a specialized financial model that could handle sensitive data and be hosted internally.
:p What is BloombergGPT?
??x
BloombergGPT is a mid-sized language model with 50 billion parameters, developed by Bloomberg in March 2023 for handling financial tasks and managing sensitive data. It required extensive GPU training time and significant cost to develop.
x??

---

#### Training Cost of BloombergGPT
The training process for BloombergGPT consumed 1.3 million A100 GPU hours and had an estimated cost between $1.3 million and$2.6 million, excluding data costs.
:p What was the cost of training BloombergGPT?
??x
The training of BloombergGPT required 1.3 million A100 GPU hours and cost between $1.3 million and$2.6 million (excluding data). This high investment reflects the significant resources needed for specialized model development.
x??

---

#### GPT-4 Performance Compared to BloombergGPT
In March 2023, OpenAI released GPT-4-0314, which outperformed BloombergGPT in various financial benchmarks. A key benchmark was FiQA sentiment analysis and ConvFinQA accuracy.
:p How did GPT-4 compare to BloombergGPT?
??x
GPT-4-0314 significantly outperformed BloombergGPT across multiple financial benchmarks, particularly in tasks like FiQA sentiment analysis (87.15 weighted F1) and ConvFinQA accuracy (76.48). BloombergGPT performed worse with 75.07 for FiQA sentiment analysis and 43.41 for ConvFinQA.
x??

---

#### Mid-Size Models Competing with GPT-4
Since the release of GPT-4, several mid-size models have been developed to compete in financial domains. These include Claude 3.5 Sonnet (70B parameters), Llama 3-70B-Instruct , and Qwen2-72B-Instruct . These are open-source and self-hostable.
:p What mid-size models competed with GPT-4?
??x
Mid-size models like Claude 3.5 Sonnet (70 billion parameters), Llama 3-70B-Instruct, and Qwen2-72B-Instruct have emerged to compete with GPT-4 in financial tasks. These models are open-source, allowing for self-hosting.
x??

---

#### Finetuning vs. Prompt Tuning
Finetuning a model can optimize token usage compared to using long prompts. This is because you can use many more examples during finetuning without hitting context length limits.
:p Why is finetuning beneficial?
??x
Finetuning is beneficial as it allows the use of a larger number of training examples, thus optimizing token usage and reducing latency and cost associated with lengthy prompts. Unlike prompt tuning, where input tokens increase with example inclusion, finetuning can handle more examples within model parameters.
x??

---

#### Example of Finetuning
Using BloombergGPT as an example, developers could fine-tune the model on specific financial data to improve performance for their use cases without needing long, token-heavy prompts.
:p What is an example of finetuning?
??x
Finetuning can be exemplified by using a specialized dataset like financial transactions or financial reports to train BloombergGPT. This would allow developers to use shorter, more efficient prompts while maintaining high performance on specific tasks.
x??

---

#### Context Length Limitations
Even with finetuning, the number of examples you can include is limited by the model's maximum context length. However, there’s no limit compared to prompt tuning.
:p What are the limitations of using many examples in a prompt?
??x
The use of many examples in a prompt is constrained by the model's context length, limiting input token usage. Finetuning bypasses this limitation but still adheres to the maximum context length during inference or interaction.
x??

---

#### Information-Based Failures vs. Behavioral Issues
Background context: The passage discusses how to decide between RAG (Retrieval-Augmented Generation) and finetuning based on the nature of the model's failures—whether they are information-based or behavioral issues.

:p How can you determine if a model's failure is due to a lack of information versus behavioral issues?
??x
If the model's output is factually incorrect or outdated, it suggests an information-based failure. For example, asking about current events where the answer provided by the model is out-of-date indicates that the model lacks up-to-date information.

On the other hand, if the output is factually correct but irrelevant to the task or does not follow expected formats (like HTML code), this points towards behavioral issues.
x??

---

#### RAG vs. Finetuning for Information-Based Failures
Background context: The passage highlights that when a model fails due to lacking information, using RAG can be more effective than finetuning.

:p In which scenario would you prefer RAG over finetuning?
??x
You should prefer RAG over finetuning when the model's failures are due to a lack of up-to-date or relevant information. For example, if asking about current events results in outdated answers, RAG can provide the necessary context and data from external sources.
x??

---

#### Behavioral Issues Requiring Finetuning
Background context: The passage explains that finetuning is useful when models have behavioral issues such as providing irrelevant outputs or not following expected formats.

:p What are some examples of behavioral issues in model outputs?
??x
Examples include:
- Providing accurate but incomplete technical specifications for a software project.
- Generating HTML code that does not compile, indicating the model lacks exposure to proper syntax during training.
x??

---

#### RAG Performance vs. Finetuning
Background context: The passage provides experimental data comparing RAG and finetuning performance on tasks requiring up-to-date information.

:p According to Ovadia et al.'s (2024) research, when did RAG outperform finetuned models?
??x
According to the study by Ovadia et al. (2024), for tasks that require up-to-date information such as questions about current events, RAG outperformed finetuned models.
x??

---

#### Combining RAG and Finetuning
Background context: The passage suggests that combining RAG with finetuning can sometimes enhance model performance.

:p How can you leverage both RAG and finetuning in a model adaptation strategy?
??x
You can start by addressing information-based failures using RAG, which provides the model access to relevant external sources. For behavioral issues such as irrelevant outputs or incorrect formats, consider finetuning with targeted data. Combining both approaches might offer a balanced solution depending on the specific task and failure modes.
x??

---

#### Application Development Flow
Background context: The passage illustrates potential paths an application development process may follow, starting from simple retrieval methods to more complex techniques like RAG and finetuning.

:p What are some possible steps in adapting a model for a new task?
??x
Possible steps include:
1. Simple term-based retrieval (e.g., using BM25).
2. Experimenting with hybrid search or more complex retrieval systems.
3. Finetuning the model on relevant data.
4. Combining RAG and finetuning to leverage both strengths.

This flexible approach allows for addressing different types of failures in models, ensuring robust performance across various tasks.
x??

---

