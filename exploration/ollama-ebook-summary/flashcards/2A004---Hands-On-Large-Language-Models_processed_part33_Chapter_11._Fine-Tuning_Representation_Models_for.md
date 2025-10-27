# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 33)

**Starting Chapter:** Chapter 11. Fine-Tuning Representation Models for Classification. Supervised Classification

---

#### Fine-Tuning Representation Models for Classification

Background context: In this chapter, we explore how to fine-tune pretrained models specifically for classification tasks. This process can lead to better performance compared to using a frozen model. Fine-tuning involves updating both the model parameters and the classification head during training.

:p What is fine-tuning in the context of representation models for classification?
??x
Fine-tuning refers to the process of adapting a pretrained model's parameters, particularly its last layers, along with a new classification head, on a specific task dataset. This allows the model to learn more task-specific features and potentially improve performance.
x??

---
#### Supervised Classification

Background context: In previous chapters, we used pretrained models that were kept frozen (non-trainable) for classification tasks. However, fine-tuning these models can lead to better results if we have enough data.

:p What is the difference between using a frozen model and fine-tuning in supervised classification?
??x
Using a frozen model means keeping the pretrained model's parameters fixed while training only the new classification head on task-specific data. Fine-tuning, however, updates both the model parameters and the classification head, allowing for more task-specific learning.
x??

---
#### SetFit Method

Background context: SetFit is an efficient method for fine-tuning a high-performing model using a small number of training examples.

:p What is SetFit and how does it differ from traditional fine-tuning methods?
??x
SetFit is a method that uses a simple and efficient approach to fine-tune pretrained models with very few labeled examples. It differs from traditional fine-tuning by focusing on minimal modifications, typically limited to the classification head, rather than extensive retraining of the entire model.
x??

---
#### Continued Pretraining

Background context: Continuing pretraining involves using masked language modeling (MLM) to further train a pretrained model.

:p What does continued pretraining involve in the context of BERT models?
??x
Continued pretraining with BERT involves continuing the training process by applying masked language modeling (MLM), where some tokens are randomly masked, and the model is trained to predict these tokens. This helps in refining the learned representations.
x??

---
#### Named-Entity Recognition

Background context: Named-Entity Recognition (NER) is a task that involves identifying named entities such as people, organizations, locations, etc., in text.

:p How does fine-tuning BERT for token-level classification differ from other tasks?
??x
Fine-tuning BERT for token-level classification, like NER, focuses on recognizing and classifying each token individually. This differs from other tasks where the entire input is classified as a whole.
x??

---

#### Fine-Tuning a Pretrained BERT Model
Fine-tuning a pretrained model like BERT involves adapting it to a specific task by training both the pretrained layers and any newly added layers. This process is crucial for improving the model's performance on specific tasks without losing the general knowledge gained from the initial pretraining.
:p What dataset was used for fine-tuning the model in this example?
??x
The Rotten Tomatoes dataset, which contains 5,331 positive and 5,331 negative movie reviews, was used for fine-tuning. This dataset is suitable for binary classification tasks such as sentiment analysis.
x??

---
#### Selecting the Model
Choosing a pretrained model that has been trained on relevant data (like BERT) can significantly improve performance in downstream tasks. The `bert-base-cased` model was selected here because it was pretrained on English Wikipedia and a large dataset of unpublished books, making it suitable for text classification.
:p Which model was chosen for the fine-tuning process?
??x
The "bert-base-cased" model was chosen for the fine-tuning process. This model was pretrained on the English Wikipedia and a large dataset consisting of unpublished books, which made it appropriate for text classification tasks like sentiment analysis.
x??

---
#### Tokenizing Data
Tokenization is essential to prepare input data for models that require specific input formats (like BERT). The `AutoTokenizer` class from Hugging Face's Transformers library was used to tokenize the data. This process involves breaking down text into smaller tokens and handling padding to ensure uniform input size.
:p How was the data tokenized in this example?
??x
The data was tokenized using the `AutoTokenizer` class from the Transformers library. The `preprocess_function` defined a method for tokenizing the text by passing it through the tokenizer with truncation enabled to handle varying lengths of input sequences.

Code Example:
```python
from transformers import AutoTokenizer

model_id = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_id)

def preprocess_function(examples):
    """Tokenize input data"""
    return tokenizer(examples["text"], truncation=True)
```
x??

---
#### Defining Metrics for Evaluation
Defining evaluation metrics is crucial to measure the model's performance during training. The F1 score was chosen as it provides a balanced measure of precision and recall, which are important in sentiment analysis tasks.
:p Which metric was used for evaluating the model?
??x
The F1 score was used for evaluating the model. This metric provides a balance between precision and recall, making it suitable for imbalanced datasets like sentiment analysis where both false positives and false negatives are costly.

Code Example:
```python
import numpy as np
from datasets import load_metric

def compute_metrics(eval_pred):
    """Calculate F1 score"""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    load_f1 = load_metric("f1")
    f1 = load_f1.compute(predictions=predictions, references=labels)["f1"]
    return {"f1": f1}
```
x??

---
#### Setting Up the Trainer
The `Trainer` class from Hugging Face's Transformers library was used to execute the training process. It handles the training loop, logging, and evaluation of the model.
:p What is the role of the `Trainer` in this context?
??x
The `Trainer` class plays a crucial role by handling the entire training process, including managing the training and evaluation loops, saving checkpoints, and logging metrics. In this context, it was used to train the fine-tuned BERT model on the Rotten Tomatoes dataset.

Code Example:
```python
from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    "model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)
```
x??

---

#### F1 Score and Model Performance Comparison
Background context explaining the concept. In this scenario, we compare the performance of a fine-tuned model with that of a task-specific pretrained model. The fine-tuned model achieved an F1 score of 0.85, which is higher than the 0.80 obtained by the task-specific pretrained model in Chapter 4.

:p What was the difference between the F1 scores of the fine-tuned model and the task-specific model?
??x
The fine-tuned model's F1 score of 0.85 was notably higher (0.05) than that of the task-specific model, which had an F1 score of 0.80.
x??

---

#### Freezing Layers in BERT Model
Background context explaining the concept. This section discusses how to fine-tune a BERT model by selectively freezing certain layers, specifically focusing on the encoder blocks and embedding layers.

:p How can we freeze specific layers in a BERT model for fine-tuning?
??x
To freeze specific layers in a BERT model, you need to iterate through the model's parameters and set their `requires_grad` attribute to False. For instance, if we want to train only the classification head and freeze all other layers:

```python
for name, param in model.named_parameters():
    # Trainable classification head
    if name.startswith("classifier"):
        param.requires_grad = True
    else:
        param.requires_grad = False
```

This code snippet ensures that only the `classifier` layer can be updated during training while all other layers are frozen.

x??

---

#### Fine-Tuning with Fully Frozen Encoder Blocks
Background context explaining the concept. In this scenario, we fully freeze all encoder blocks and embedding layers of a BERT model, leaving only the classification head for fine-tuning.

:p What did the results show when we fully froze all encoder blocks in the BERT model?
??x
When we fully froze all encoder blocks and left only the classification head for fine-tuning, the F1 score decreased significantly. The evaluation resulted in an F1 score of 0.63 compared to the original 0.85 score.

This outcome illustrates that while freezing many layers can speed up training, it may lead to suboptimal performance if too many layers are frozen and not enough information is passed through for learning new representations.

x??

---

#### Fine-Tuning with Partially Frozen Encoder Blocks
Background context explaining the concept. This section explores the impact of partially freezing encoder blocks by keeping some blocks trainable while freezing others, providing a balance between computational efficiency and model performance.

:p What was the effect on the F1 score when we froze the first 10 encoder blocks in BERT?
??x
Freezing the first 10 encoder blocks and training only the remaining ones resulted in an F1 score of 0.8. This is a significant improvement compared to the previous scenario where freezing all but the classification head yielded an F1 score of 0.63.

This example shows that by strategically freezing certain layers, we can maintain high performance while reducing computational load.

x??

---

#### Iterative Fine-Tuning with Frozen Encoder Blocks
Background context explaining the concept. The final section examines how iteratively freezing encoder blocks affects model performance, demonstrating that training a subset of blocks (in this case, the first five) can achieve near-optimal results.

:p What did the iterative fine-tuning approach reveal about the number of encoder blocks to freeze?
??x
The iterative fine-tuning approach revealed that training only the first five encoder blocks was sufficient to nearly match the performance of training all 12 blocks. This suggests that in scenarios with limited computational resources, it may be more efficient to train a subset of layers while still achieving good results.

x??

---

