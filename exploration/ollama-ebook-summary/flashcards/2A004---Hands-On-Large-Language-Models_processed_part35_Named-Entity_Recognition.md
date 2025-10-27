# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 35)

**Starting Chapter:** Named-Entity Recognition

---

#### Fine-Tuning Model for Classification

Background context: The provided text discusses fine-tuning a pretrained BERT model on specific classification tasks, such as predicting movie reviews or named-entity recognition (NER). The focus is on leveraging the existing pretraining to adapt the model to new tasks.

:p What is the purpose of fine-tuning a BERT model for classification?
??x
The purpose is to adapt a pretrained BERT model to perform well on specific classification tasks, such as movie reviews or named entities. This involves retraining the model using task-specific data to optimize its performance on these new tasks.
x??

---

#### Predicting Movie Reviews

Background context: The text demonstrates how to fine-tune a BERT model for predicting movie reviews by filling in masked words and observing the bias towards the training data.

:p What did the example predict when fine-tuning the BERT model?
??x
The example predicted sentences like "What a horrible movie," "What a horrible film," "What a horrible mess," and "What a horrible comedy," showing that the model is biased towards its training data.
x??

---

#### Fine-Tuning for Named-Entity Recognition (NER)

Background context: The text explains how to fine-tune a BERT model specifically for NER, which involves classifying individual tokens or words rather than entire documents. This is useful for tasks like de-identification and anonymization.

:p What does NER stand for, and what are its applications?
??x
NER stands for Named-Entity Recognition. Its applications include identifying entities such as people, locations, organizations, dates, etc., in text. These can be useful in tasks like de-identification or anonymization where sensitive data needs to be handled.
x??

---

#### Preprocessing for NER

Background context: The text mentions that fine-tuning a BERT model for NER requires preprocessing the data at a token level rather than document level, which is different from how it was done in previous examples.

:p How does the preprocessing differ when fine-tuning BERT for NER compared to other tasks?
??x
In fine-tuning BERT for NER, the preprocessing involves handling individual tokens or words instead of entire documents. This means each token needs to be classified individually, focusing on detecting named entities such as people and locations.
x??

---

#### Token-Level Classification

Background context: The text emphasizes that in NER tasks, the model makes predictions at a token level rather than aggregating embeddings for entire sentences.

:p What is unique about token-level classification in BERT?
??x
Token-level classification in BERT involves making predictions on individual tokens instead of relying on aggregated or pooled token embeddings. This approach focuses on classifying each word or subword unit, which is crucial for tasks like NER where specific entities need to be identified.
x??

---

#### Fine-Tuning Process for NER

Background context: The text outlines the process of fine-tuning a BERT model for NER, highlighting the differences in how data is processed and classified compared to previous classification tasks.

:p What steps are involved in fine-tuning a BERT model for NER?
??x
The steps involve loading the pretrained BERT model, specifying the number of labels (e.g., 2 for binary classification), and using a tokenizer from the same pretraining. The data is then processed to classify individual tokens, focusing on named entities such as people or locations.
x??

---

#### Fine-Tuning with Named-Entities

Background context: The text describes the fine-tuning process where BERT is adapted to recognize specific named entities in the text.

:p How does the fine-tuned BERT model handle named entities during classification?
??x
The fine-tuned BERT model classifies individual tokens to identify and label named entities, such as people or locations. This allows for accurate detection of these entities within the text.
x??

---

#### CoNLL-2003 Dataset Overview
The CoNLL-2003 dataset is used for Named Entity Recognition (NER) tasks and contains various types of named entities such as person, organization, location, miscellaneous, and no entity. It includes approximately 14,000 training samples.
:p What does the CoNLL-2003 dataset contain?
??x
The dataset contains sentences with different named entities like people, organizations, locations, and miscellaneous categories. Each sentence is annotated at the word level to identify these entities.
x??

---

#### Dataset Structure in Example
The provided example demonstrates how the dataset structure looks for a specific example from the CoNLL-2003 dataset. It includes keys such as `tokens`, `pos_tags`, `chunk_tags`, and `ner_tags`.
:p How is the data structured in an individual example of the CoNLL-2003 dataset?
??x
In each example, the dataset provides information including tokens (words), their part-of-speech tags (`pos_tags`), chunk tags for syntactic structure, and named entity recognition tags (`ner_tags`). For instance:
```python
example = {
    'id': '848',
    'tokens': ['Dean', 'Palmer', 'hit', 'his', '30th', 'homer', 'for', 'the', 'Rangers', '.'],
    'pos_tags': [22, 22, 38, 29, 16, 21, 15, 12, 23, 7],
    'chunk_tags': [11, 12, 21, 11, 12, 12, 13, 11, 12, 0],
    'ner_tags': [1, 2, 0, 0, 0, 0, 0, 0, 3, 0]
}
```
x??

---

#### Named Entity Tags
The `ner_tags` key in the dataset provides labels for each word indicating their corresponding named entity type. These include person (PER), organization (ORG), location (LOC), miscellaneous entities (MISC), and no entity (O).
:p What do the tags in the `ner_tags` field signify?
??x
The `ner_tags` field uses specific tags like 'O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC', and 'B-MISC' to denote named entities. For instance:
```python
label2id = {
    "O": 0,
    "B-PER": 1,
    "I-PER": 2,
    "B-ORG": 3,
    "I-ORG": 4,
    "B-LOC": 5,
    "I-LOC": 6,
    "B-MISC": 7,
    "I-MISC": 8
}
```
x??

---

#### Tokenization and Sub-tokenization
The process of tokenizing sentences in the dataset to sub-tokens is necessary for further analysis. The example provided uses BERT's tokenizer which splits words like 'homer' into `home` and `##r`.
:p How does the BERT tokenizer handle word splitting?
??x
BERT tokens some words into sub-tokens, especially if they are part of a larger term. For instance:
```python
token_ids = tokenizer(example["tokens"], is_split_into_words=True)["input_ids"]
sub_tokens = tokenizer.convert_ids_to_tokens(token_ids)
sub_tokens 
```
This results in output like `['[CLS]', 'Dean', 'Palmer', 'hit', 'his', '30th', 'home', '##r', 'for', 'the', 'Rangers', '.', '[SEP]']`.

To handle this, a function is needed to align the labels with these sub-tokenized tokens.
x??

---

#### Label Alignment for Tokenized Input
The `align_labels` function ensures that each token gets the correct label based on whether it's part of an entity or not. It handles splitting and tagging correctly during tokenization.
:p How does the `align_labels` function ensure correct labeling for sub-tokenized entities?
??x
The `align_labels` function processes input examples by first tokenizing them, then aligning their labels with the resulting sub-tokens. Here's an example of how it works:
```python
def align_labels(examples):
    # Tokenize input and get word_ids to map tokens back to words
    token_ids = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = examples["ner_tags"]
    updated_labels = []
    for index, label in enumerate(labels):
        word_ids = token_ids.word_ids(batch_index=index)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx == previous_word_idx:
                # Start of a new word
                previous_word_idx = word_idx
                updated_label = -100 if word_idx is None else label[word_idx]
                label_ids.append(updated_label)
            elif word_idx is None:
                label_ids.append(-100)
            else:
                updated_label = label[word_idx]
                # If the label is B-XXX we change it to I-XXX
                if updated_label % 2 == 1:
                    updated_label += 1
                label_ids.append(updated_label)
        updated_labels.append(label_ids)
    token_ids["labels"] = updated_labels
    return token_ids

tokenized = dataset.map(align_labels, batched=True)
```
This ensures that the labels are correctly aligned and sub-tokenized words get appropriate tagging.
x??

---

#### Evaluation Metrics for Token-Level NER
The evaluation metrics need to consider each token's predicted label compared to its true label. The `compute_metrics` function uses Hugging Face’s `seqeval` package to handle this at a token level.
:p How is the evaluation done for token-level Named Entity Recognition tasks?
??x
For token-level NER, we use a custom `compute_metrics` function that processes predictions and labels on a per-token basis. Here's an example of how it works:
```python
import evaluate

seqeval = evaluate.load("seqeval")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=2)
    true_predictions = []
    true_labels = []

    for prediction, label in zip(predictions, labels):
        for token_prediction, token_label in zip(prediction, label):
            if token_label == -100:
                true_predictions.append([id2label[token_prediction]])
                true_labels.append([id2label[token_label]])

# Usage
compute_metrics(eval_pred)
```
This function ensures that special tokens are ignored and correct labels are assigned based on the token's position within a phrase.
x??

---

#### Fine-Tuning BERT for Named-Entity Recognition (NER)
Background context: This section discusses fine-tuning a BERT model specifically for Named-Entity Recognition (NER), where the goal is to identify and classify named entities such as names, places, or organizations within unstructured text. The process involves using token-level classification rather than document-level classification.

:p What are the steps involved in fine-tuning a BERT model for NER?
??x
The steps involve several key components:

1. **Loading Pretrained Model and Tokenizer**: Load a pretrained BERT model and tokenizer.
2. **Tokenization**: Tokenize the dataset into input IDs, attention masks, and labels (true_predictions and true_labels).
3. **Creating Datasets**: Create PyTorch datasets for training and evaluation.
4. **Defining Training Arguments**: Define arguments such as learning rate, batch size, number of epochs, etc.
5. **Initializing Trainer**: Initialize a `Trainer` with the model, dataset, data_collator, and compute_metrics function.
6. **Training the Model**: Train the model using the `trainer.train()` method.
7. **Evaluating the Model**: Evaluate the model on the test set using `trainer.evaluate()`.
8. **Saving and Using the Model**: Save the fine-tuned model and use it in a pipeline for inference.

Code example:
```python
from transformers import BertTokenizer, BertForTokenClassification

# Load pretrained model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased')

# Tokenize the dataset
tokenized = ...  # Assume tokenization has been done here

# Training arguments
training_args = TrainingArguments(
    "ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)

# Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Evaluate the model
results = trainer.evaluate()
print(results)
```

x??

---

#### Computing Metrics for NER
Background context: After fine-tuning a BERT model, it is essential to evaluate its performance using appropriate metrics. The `seqeval` library can be used to compute various metrics such as precision, recall, F1-score.

:p What function computes the sequence evaluation and returns an overall F1 score?
??x
The function `seqeval.compute` is used to compute sequence evaluation metrics like precision, recall, and F1-score. It takes predicted labels (`predictions`) and reference labels (`references`) as inputs and returns a dictionary with various metrics.

Code example:
```python
from seqeval.metrics import f1_score

# Example usage of seqeval.compute
results = seqeval.compute(
    predictions=true_predictions,
    references=true_labels
)
f1_score_result = results["overall_f1"]
```

x??

---

#### DataCollatorForTokenClassification vs. DataCollatorWithPadding
Background context: For fine-tuning a BERT model, using `DataCollatorForTokenClassification` is crucial because it handles token-level classification tasks differently from text padding.

:p Why should we use `DataCollatorForTokenClassification` over `DataCollatorWithPadding` in NER?
??x
We should use `DataCollatorForTokenClassification` because it is designed to work with token-level classification tasks, which are common in Named-Entity Recognition (NER). Unlike `DataCollatorWithPadding`, which pads input sequences to the same length for batch processing, `DataCollatorForTokenClassification` handles token-level labels correctly.

Code example:
```python
from transformers import DataCollatorForTokenClassification

# Example usage of DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
```

x??

---

#### Fine-Tuning Arguments for NER
Background context: Fine-tuning a model involves defining and tuning various parameters such as learning rate, batch size, number of epochs, etc. These are encapsulated in `TrainingArguments`.

:p What are the key arguments defined in the `TrainingArguments` for fine-tuning BERT for NER?
??x
The key arguments defined in the `TrainingArguments` include:

- **Learning Rate (`learning_rate`)**: Controls how much to update the model parameters per batch.
- **Batch Size (`per_device_train_batch_size`, `per_device_eval_batch_size`)**: The number of samples used for each training and evaluation step.
- **Number of Epochs (`num_train_epochs`)**: The number of times the entire dataset will be passed through the model during training.
- **Weight Decay (`weight_decay`)**: A regularization term that penalizes large weights to prevent overfitting.
- **Save Strategy (`save_strategy`)**: Determines when and how often the model should be saved during training.
- **Report To (`report_to`)**: Where to log the training metrics.

Code example:
```python
training_args = TrainingArguments(
    "ner_model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    save_strategy="epoch",
    report_to="none"
)
```

x??

---

#### Inference with Fine-Tuned NER Model
Background context: After fine-tuning, the model can be used for inference using a pipeline.

:p How do you create and use a token classification pipeline to perform inference on a text?
??x
To create and use a token classification pipeline for performing inference, follow these steps:

1. **Save the Fine-Tuned Model**: Use `trainer.save_model` to save the fine-tuned model.
2. **Initialize Pipeline**: Use the saved model in a `pipeline`.
3. **Run Inference**: Pass text through the pipeline to get predictions.

Code example:
```python
from transformers import pipeline

# Save the fine-tuned model
trainer.save_model("ner_model")

# Initialize token classification pipeline
token_classifier = pipeline(
    "token-classification",
    model="ner_model"
)

# Run inference on a sentence
input_text = "My name is Maarten."
predictions = token_classifier(input_text)
print(predictions)
```

x??

---

