# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 31)

**Starting Chapter:** Creating an Embedding Model. Train Model

---

#### Generating Contrastive Examples
Background context: To perform contrastive learning, we need data that consists of similar/dissimilar pairs. Natural Language Inference (NLI) datasets are commonly used to generate these examples since they provide a structure where entailment, contradiction, and neutral relationships between sentences can be leveraged.

Example:
- Premise: "He is in the cinema watching Coco."
- Hypothesis 1 (Contradiction): "He is watching Frozen at home."
- Hypothesis 2 (Entailment): "In the movie theater he is watching the Disney movie Coco."

:p What are contrastive examples used for in contrastive learning?
??x
Contrastive examples are used to train models by providing pairs of sentences where one sentence is similar or dissimilar to the other. The model learns to distinguish between these pairs, optimizing its ability to understand and represent semantic similarities.
x??

---

#### Using NLI Datasets for Contrastive Learning
Background context: The General Language Understanding Evaluation (GLUE) benchmark includes datasets like Multi-Genre Natural Language Inference (MNLI), which consists of sentence pairs annotated with entailment, contradiction, or neutral relationships. These annotations help in generating positive and negative examples suitable for contrastive learning.

:p How can NLI datasets be used to generate contrastive examples?
??x
NLI datasets can be used by selecting pairs of sentences that are annotated as either entailments (positive examples) or contradictions (negative examples). This helps in creating a structured set of data where the model can learn from both similar and dissimilar sentence pairs.
x??

---

#### Training an Embedding Model
Background context: To train an embedding model, we typically use existing models like BERT and fine-tune them for specific tasks. In this example, a base BERT model is used as the word embedding layer, and a loss function is defined to optimize the model's performance.

:p How do you create a training dataset from the GLUE benchmark?
??x
The dataset can be created by loading the MNLI corpus from the GLUE benchmark and selecting a subset of annotated sentence pairs. For instance, we select 50,000 samples using `load_dataset` and `select(range(50_000))`.

```python
from datasets import load_dataset

train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")
```

This ensures we have a manageable dataset for training purposes.
x??

---

#### Defining the Loss Function
Background context: The loss function is crucial in optimizing the embedding model. In this example, we use softmax loss to optimize the model based on the similarity between sentences.

:p What is the purpose of defining a loss function in contrastive learning?
??x
The loss function helps in guiding the training process by quantifying how well the model's predictions match the ground truth. For instance, using softmax loss allows us to define a cost that the model tries to minimize during training.

```python
from sentence_transformers import losses

train_loss = losses.SoftmaxLoss(
    model=embedding_model,
    sentence_embedding_dimension=embedding_model.get_sentence_embedding_dimension(),
    num_labels=3
)
```

This setup ensures the model learns to correctly classify the relationships between sentences.
x??

---

#### Creating Evaluation Evaluator
Background context: During training, it is essential to evaluate the model's performance regularly. The Semantic Textual Similarity Benchmark (STS) is used here as an evaluator. It provides a set of sentence pairs with similarity scores.

:p How do you create an evaluator for semantic textual similarity?
??x
An evaluator can be created by using the `EmbeddingSimilarityEvaluator` from `sentence_transformers.evaluation`. This evaluator processes the validation data and calculates the cosine similarity between sentences.

```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score / 5 for score in val_sts["label"]],
    main_similarity="cosine",
)
```

This setup helps ensure the model performs well on tasks requiring semantic similarity.
x??

---

#### Training Arguments
Background context: The `SentenceTransformerTrainingArguments` define various parameters necessary for training, such as the number of epochs, batch size, and learning rate.

:p What are some important arguments in SentenceTransformerTrainingArguments?
??x
Important arguments include:
- `num_train_epochs`: Number of training epochs. A value of 1 is used here for faster training but can be increased.
- `per_device_train_batch_size` and `per_device_eval_batch_size`: Batch sizes for training and evaluation, which affect the speed of both processes.

```python
from sentence_transformers.training_args import SentenceTransformerTrainingArguments

args = SentenceTransformerTrainingArguments(
    output_dir="base_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)
```

These settings help in balancing the speed and quality of the training process.
x??

---

#### Training the Embedding Model
Background context: The `SentenceTransformerTrainer` class is used to train the embedding model using the defined arguments, dataset, loss function, and evaluator.

:p How do you start training an embedding model?
??x
Training can be started by creating a `SentenceTransformerTrainer` instance and calling its `train()` method.

```python
from sentence_transformers.trainer import SentenceTransformerTrainer

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)

trainer.train()
```

This process involves setting up the trainer with all necessary components and initiating training.
x??

#### Cosine Similarity Evaluation
Background context explaining the concept. The `pearson_cosine` value is a measure of similarity between two vectors, with higher values indicating greater similarity. It ranges from 0 to 1.
:p What is the `pearson_cosine` metric used for evaluating text embedding models?
??x
The `pearson_cosine` metric evaluates how similar two vectors are by measuring their cosine similarity. A value closer to 1 indicates a high degree of similarity between the centered vectors.
x??

---

#### Batch Size Impact on Embedding Models
Explanation about batch size impact and MNR loss. Larger batch sizes increase model difficulty as more negative rankings need to be considered, potentially improving performance.
:p How does batch size affect multiple negative ranking (MNR) loss during training?
??x
Larger batch sizes make the task of finding the best matching sentence from a set more challenging. This is because MNR loss involves ranking multiple negative samples, which increases model complexity and can improve its ability to generalize.
x??

---

#### In-Depth Evaluation Using MTEB
Description of the Massive Text Embedding Benchmark (MTEB) and its tasks. The benchmark spans 8 embedding tasks covering 58 datasets and 112 languages.
:p What is the purpose of the Massive Text Embedding Benchmark (MTEB)?
??x
The MTEB aims to provide a unified evaluation framework for comparing different text embedding models across various tasks, datasets, and languages. It includes 8 embedding tasks that cover 58 datasets and 112 languages.
x??

---

#### Evaluation Metrics for Specific Tasks
Explanation of the metrics provided by evaluating an embedding model on specific tasks like `Banking77Classification`.
:p What evaluation metrics are provided when running a specific task using MTEB?
??x
When running a specific task with MTEB, various metrics such as accuracy and F1 score are provided. These metrics help in understanding the performance of the embedding model on that particular task.
For example, `Banking77Classification` provides:
- Accuracy: 0.4926
- F1 Score: 0.4908
x??

---

#### Time Efficiency and Model Selection
Explanation about the importance of both accuracy and latency in choosing an embedding model, especially for tasks requiring fast inference.
:p Why is it important to consider both accuracy and latency when selecting an embedding model?
??x
It is crucial to balance both accuracy and latency when selecting an embedding model because real-world applications often need models that perform quickly. For instance, semantic search requires efficient inference times, making low-latency models more desirable despite potentially lower accuracy.
x??

---

#### STSB Benchmark as Example
Explanation about using the Semantic Textual Similarity Benchmark (STSB) for illustrative purposes due to time constraints.
:p Why is the STSB benchmark used instead of MTEB in this chapter?
??x
The STSB benchmark is used instead of MTEB because testing on the entire MTEB can take several hours, depending on GPU resources. The STSB provides a more practical and faster way to illustrate concepts without extensive computational overhead.
x??

---

#### Restarting the Notebook
Background context explaining why restarting the notebook is important. Clear VRAM and ensure a clean slate for subsequent training sessions.

:p Why is it important to restart the notebook after training and evaluating a model?
??x
Restarting the notebook ensures that all VRAM is cleared, which helps in managing memory efficiently for the next set of training examples or tasks.
x??

---

#### Softmax Loss Function
Background context on softmax loss function, explaining its usage and limitations.

:p What is the issue with using softmax loss for training sentence-transformer models?
??x
Softmax loss is generally not advised because there are more performant loss functions available. Softmax loss can lead to suboptimal performance compared to other losses like cosine similarity or multiple negatives ranking (MNR) loss.
x??

---

#### Cosine Similarity Loss
Explanation of the cosine similarity loss and its application in semantic textual similarity tasks.

:p How does the cosine similarity loss work?
??x
The cosine similarity loss minimizes the cosine distance between semantically similar sentences and maximizes the distance between semantically dissimilar ones. It calculates the cosine similarity between two embeddings and compares it to a labeled similarity score, allowing the model to learn the degree of sentence similarity.
x??

---

#### Converting Labels for Cosine Similarity
Explanation on how to convert labels (entailment, neutral, contradiction) into a 0-1 scale.

:p How do you convert entailment, neutral, and contradiction labels to values between 0 and 1?
??x
Entailment is given a label of 1 because it represents high similarity. Neutral and contradiction are labeled as 0 since they represent dissimilarity.
```python
from datasets import Dataset

# Define the mapping for entailment, neutral, and contradiction labels
mapping = {2: 0, 1: 0, 0: 1}

# Apply the mapping to the dataset labels
train_dataset = train_dataset.map(
    lambda sample: {"label": float(mapping[sample["label"]])},
    remove_columns="label"
)
```
x??

---

#### Creating an Embedding Model with Cosine Similarity Loss
Explanation on setting up a model and loss function for training with cosine similarity.

:p How do you set up the model and loss function for training using cosine similarity?
??x
First, create your dataset and evaluator:
```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Load validation dataset for STS
val_sts = load_dataset("glue", "stsb", split="validation")

# Create an embedding similarity evaluator
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score / 5 for score in val_sts["label"]],
    main_similarity="cosine"
)
```
Then, define the model and loss function:
```python
from sentence_transformers import SentenceTransformer, losses

# Define the model
embedding_model = SentenceTransformer("bert-base-uncased")

# Define the loss function using cosine similarity
train_loss = losses.CosineSimilarityLoss(model=embedding_model)

# Set up training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="cosineloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32
)
```
x??

---

#### Efficient Natural Language Response Suggestion for Smart Reply
Background context explaining the concept. The paper discusses an efficient method for suggesting natural language responses to smart reply systems, using a specific training approach with predefined parameters.

:p What is the purpose of the `warmup_steps`, `fp16`, `eval_steps`, and `logging_steps` in the model training process?
??x
The `warmup_steps` parameter initializes the model by allowing it to adjust its weights over 100 steps. The `fp16` flag enables mixed precision training, which can speed up training on hardware that supports half-precision arithmetic. `eval_steps` and `logging_steps` both set intervals at which the model is evaluated and logging information is recorded during training.

```python
# Example setup for a trainer in Hugging Face's SentenceTransformer library
args = TrainingArguments(
    warmup_steps=100,
    fp16=True,  # Enables mixed precision training
    eval_steps=100,
    logging_steps=100
)

trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
```
x??

---

#### Evaluating the Model Post-Training
The text describes evaluating a trained model using an `evaluator`, which provides various similarity scores between different types of sentence pairs.

:p What score does the model achieve post-training, and how does it compare to the softmax loss example?
??x
The model achieves a Pearson cosine score of 0.7222322163831805, which is significantly better than the softmax loss example that scored only 0.59. This demonstrates the effectiveness of using contrastive predictive coding in improving performance.

```python
# Evaluation code snippet to get scores after training
evaluator(embedding_model)
```
x??

---

#### Multiple Negatives Ranking (MNR) Loss
The passage explains MNR loss, which is a method for training models by leveraging positive and negative sentence pairs. It involves minimizing the distance between related sentences while maximizing the distance between unrelated ones.

:p What are in-batch negatives in the context of MNR loss?
??x
In the context of MNR loss, in-batch negatives refer to negative examples that come from within the same batch as a positive example. These negatives are constructed by combining a sentence with another unrelated sentence within the same batch, effectively creating pairs where one is related and the other is not.

```java
// Pseudocode for generating in-batch negatives
for (Sentence s1 : batch) {
    for (Sentence s2 : batch) {
        if (!s1.isSimilarTo(s2)) { // Check if sentences are unrelated
            add(s1, s2) as a negative pair;
        }
    }
}
```
x??

---

#### InfoNCE and NTXent Loss
The text refers to MNR loss often being called InfoNCE or NTXent loss. These losses use positive pairs of sentences along with in-batch negatives to train models effectively.

:p How does the multiple negatives ranking (MNR) loss work?
??x
Multiple Negatives Ranking (MNR) loss works by creating both positive and negative sentence pairs, where positive pairs are related and negative pairs are unrelated. The model is trained to minimize the distance between related sentences while maximizing the distance between unrelated ones.

The process involves:
1. Generating positive pairs of sentences.
2. Creating in-batch negatives by combining a positive pair with another positive pair or using unrelated sentences from within the same batch.
3. Calculating embeddings for each sentence and applying cosine similarity.
4. Using these similarities to classify pairs as negative or positive, treating it as a classification problem with cross-entropy loss.

```java
// Pseudocode for MNR loss implementation
public class MnrlLoss {
    public double calculateLoss(List<Pair<Sentence, Sentence>> pairs) {
        List<Double> scores = new ArrayList<>();
        for (Pair<Sentence, Sentence> pair : pairs) {
            // Calculate cosine similarity between embeddings of the two sentences in the pair
            double score = calculateCosineSimilarity(pair.getFirst(), pair.getSecond());
            scores.add(score);
        }
        // Convert similarities to log-likelihood probabilities and apply cross-entropy loss
        return crossEntropyLoss(scores, trueLabels);
    }

    private double calculateCosineSimilarity(Sentence s1, Sentence s2) {
        Embedding e1 = embedder.embed(s1);
        Embedding e2 = embedder.embed(s2);
        // Normalize embeddings and compute cosine similarity
        return dotProduct(e1.normalized(), e2.normalized());
    }

    private double crossEntropyLoss(List<Double> scores, List<Boolean> labels) {
        double loss = 0.0;
        for (int i = 0; i < scores.size(); i++) {
            if (labels.get(i)) { // Positive pair
                loss -= Math.log(scores.get(i));
            } else { // Negative pair
                loss -= Math.log(1 - scores.get(i));
            }
        }
        return loss;
    }
}
```
x??

---

#### Loading MNLI Dataset
Background context: The MNLI (Multi-Genre Natural Language Inference) dataset is used to train models for natural language inference tasks, where given a premise and a hypothesis, we determine if the hypothesis entails, contradicts, or is neutral with respect to the premise.

:p How do you load the MNLI dataset from GLUE?
??x
To load the MNLI dataset from the GLUE benchmark, use the following code snippet:
```python
from datasets import load_dataset

# Load the MNLI dataset and select a subset of 50,000 examples for training.
mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
```
x??

---

#### Preparing Data with Positive and Negative Examples
Background context: After loading the MNLI dataset, we filter it to include only sentences labeled as entailment (labeled as 0). Then, negative examples are added by randomly sampling sentences from the "hypothesis" field.

:p How do you prepare data for training with both positive and negative examples?
??x
To prepare data for training with both positive and negative examples, follow these steps:
```python
import random

# Load MNLI dataset from GLUE
mnli = load_dataset("glue", "mnli", split="train").select(range(50_000))
mnli = mnli.remove_columns("idx")

# Filter the data to include only entailment examples (label 0)
mnli = mnli.filter(lambda x: True if x["label"] == 0 else False)

# Prepare data and add a soft negative
train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)

for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)
```
x??

---

#### Creating an Embedding Model
Background context: An embedding model is created using the SentenceTransformer framework, which allows for training models to generate dense vector representations of text. These embeddings can be used in various NLP tasks such as similarity evaluation and text classification.

:p How do you create a dataset with anchor, positive, and negative examples?
??x
To create a dataset with anchor, positive, and negative examples:
```python
train_dataset = {"anchor": [], "positive": [], "negative": []}
soft_negatives = mnli["hypothesis"]
random.shuffle(soft_negatives)

for row, soft_negative in tqdm(zip(mnli, soft_negatives)):
    train_dataset["anchor"].append(row["premise"])
    train_dataset["positive"].append(row["hypothesis"])
    train_dataset["negative"].append(soft_negative)
train_dataset = Dataset.from_dict(train_dataset)
```
x??

---

#### Defining the Evaluator
Background context: The evaluator is used to measure how well the model performs on a validation set. It compares sentences and computes similarity scores.

:p How do you define an embedding similarity evaluator?
??x
To define an embedding similarity evaluator, use the following code:
```python
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator

# Load STS (Semantic Textual Similarity) dataset for validation
val_sts = load_dataset("glue", "stsb", split="validation")

# Create an evaluator to compare sentences and compute similarity scores
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score / 5 for score in val_sts["label"]],
    main_similarity="cosine",
)
```
x??

---

#### Training with MNR Loss
Background context: The Multiple Negatives Ranking (MNR) loss is used to train the model by ranking anchor, positive, and negative examples. This helps in improving the model's ability to distinguish between similar and dissimilar text pairs.

:p How do you train the model using MNR loss?
??x
To train the model using MNR loss, follow these steps:
```python
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

# Define the model embedding model = SentenceTransformer('bert-base-uncased')
# Loss function train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

# Define training arguments args = SentenceTransformerTrainingArguments(
    output_dir="mnrloss_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100
)

# Train the model trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator
)
trainer.train()
```
x??

---

#### Evaluating the Trained Model
Background context: After training, the model's performance is evaluated using the defined evaluator. This provides metrics such as Pearson and Spearman correlation coefficients for cosine similarity.

:p How do you evaluate the trained model?
??x
To evaluate the trained model:
```python
# Evaluate our trained model
evaluator(embedding_model)
```
The output will include various metrics like Pearson, Spearman correlations for different distance measures.
x??

---

#### Larger Batch Sizes and MNR Loss
Background context: When using MNR (Most Negatives Rejection) loss, larger batch sizes are generally preferred because they make the task of finding the best matching sentence more challenging. The model needs to distinguish between a higher number of potential pairs of sentences.
:p What is the benefit of using larger batch sizes with MNR loss?
??x
Larger batch sizes increase the difficulty for the model by requiring it to find the best matching sentence from a larger set, which can lead to better performance due to more nuanced learning. However, if in-batch negatives (easier negatives) are unrelated to the question, this could make the task too easy.
x??

---

#### In-Batch Negatives vs Hard Negatives
Background context: Easy negatives sampled randomly might not be related enough to the question, making the task of finding the correct answer too simple. Instead, hard negatives that are closely related but incorrect should be used for a more challenging and effective training process.
:p What is the difference between easy and hard negatives in MNR loss?
??x
Easy negatives are typically unrelated to both the question and answer. Hard negatives are very similar to the question but are generally the wrong answer, making the task more difficult for the model and improving its performance as it learns nuanced representations.
x??

---

#### Gathering Negatives: Easy Negatives
Background context: Easy negatives can be generated by randomly sampling documents. This method is simple but might not provide sufficiently challenging training examples since they could be completely unrelated to the question and answer pairs.
:p How are easy negatives typically created?
??x
Easy negatives are created through random sampling of documents. This method can lead to in-batch or "easy" negatives that may have nothing to do with the specific question and answer pair, making the task for the embedding model too simple.
x??

---

#### Gathering Negatives: Semi-Hard Negatives
Background context: Semi-hard negatives are sentences that share some similarity with the topic of the question and answer but are still unrelated. They can be found using a pretrained embedding model by applying cosine similarity on sentence embeddings to find highly related ones, though they generally don't constitute hard negatives.
:p How are semi-hard negatives identified?
??x
Semi-hard negatives are identified using a pretrained embedding model through cosine similarity analysis of sentence embeddings. This method finds sentences that are highly related but does not necessarily create question/answer pairs, thus not always constituting hard negatives.
x??

---

#### Gathering Negatives: Hard Negatives
Background context: Hard negatives are very similar to the question and answer pair but are generally incorrect answers. They require more effort to generate, either through manual labeling or using a generative model. These can significantly improve embedding model performance by making the task more challenging.
:p How can hard negatives be generated?
??x
Hard negatives can be generated manually (e.g., from semi-hard negatives) or by using a generative model. They should contain information related to the question but be incorrect answers, such as "More than a million people live in Utrecht, which is more than Amsterdam," relating to the topic of Amsterdam’s population but not being the correct answer.
x??

---

#### Fine-Tuning an Embedding Model Using Pretrained Models
Fine-tuning embedding models using pre-trained sentence-transformers models is a powerful method for adapting existing models to specific tasks or datasets. This approach allows us to leverage the knowledge gained during large-scale training and adapt it to our particular use case, often more efficiently than training from scratch.

In this process, we select a pre-trained model like `all-MiniLM-L6-v2` which is known for its performance across many domains due to its small size and efficiency. We then fine-tune this model on specific data to improve its relevance to our task.

:p How do you start the fine-tuning process using a pre-trained sentence-transformers model?
??x
The process starts by loading the necessary libraries, selecting or preparing your dataset, defining the evaluation method, choosing the appropriate model and loss function, setting up training arguments, and finally initiating the training. This is done in steps as follows:

1. **Load Libraries**:
   - Import required packages such as `datasets`, `sentence_transformers`, etc.

2. **Prepare Data**:
   - Load datasets using `load_dataset` from Hugging Face's Datasets library.
   - Create an evaluator for the dataset, which helps in evaluating the performance of your model during and after training.

3. **Define Model and Loss Function**:
   - Use a pre-trained embedding model like `all-MiniLM-L6-v2`.
   - Define the loss function suitable for the task (e.g., `MultipleNegativesRankingLoss`).

4. **Set Up Training Arguments**:
   - Create training arguments specifying details such as output directory, number of epochs, batch size, evaluation steps, etc.

5. **Train Model**:
   - Initialize a trainer with your model, loss function, dataset, and evaluator.
   - Train the model using `trainer.train()`.

Here is a simplified example:

```python
from datasets import load_dataset
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers import SentenceTransformer, losses, SentenceTransformerTrainer, SentenceTransformerTrainingArguments

# Load MNLI dataset from GLUE
train_dataset = load_dataset("glue", "mnli", split="train").select(range(50_000))
train_dataset = train_dataset.remove_columns("idx")

# Create an embedding similarity evaluator for STSB validation
val_sts = load_dataset("glue", "stsb", split="validation")
evaluator = EmbeddingSimilarityEvaluator(
    sentences1=val_sts["sentence1"],
    sentences2=val_sts["sentence2"],
    scores=[score/5 for score in val_sts["label"]],
    main_similarity="cosine",
)

# Define model and loss function
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
train_loss = losses.MultipleNegativesRankingLoss(model=embedding_model)

# Define training arguments
args = SentenceTransformerTrainingArguments(
    output_dir="finetuned_embedding_model",
    num_train_epochs=1,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    warmup_steps=100,
    fp16=True,
    eval_steps=100,
    logging_steps=100,
)

# Train model
trainer = SentenceTransformerTrainer(
    model=embedding_model,
    args=args,
    train_dataset=train_dataset,
    loss=train_loss,
    evaluator=evaluator,
)
trainer.train()
```

This example demonstrates the fine-tuning process using a pre-trained embedding model, showcasing how to set up and execute training with specific parameters.

x??

---

#### Evaluation of Fine-Tuned Embedding Model
After fine-tuning, evaluating the performance of the model is crucial. Evaluators like `EmbeddingSimilarityEvaluator` are used to measure how well the fine-tuned model performs on a validation dataset. Common metrics include Pearson and Spearman correlation coefficients for cosine similarity.

:p What evaluation method did we use after training our fine-tuned embedding model?
??x
We used `EmbeddingSimilarityEvaluator` from the sentence-transformers library to evaluate the performance of the fine-tuned embedding model. This evaluator calculates various metrics such as Pearson and Spearman correlation coefficients, which are commonly used for measuring the similarity between embeddings.

Here is an example of how we evaluated our trained model:

```python
# Evaluate our trained model
evaluator(embedding_model)
```

The output provided the following scores:
- `pearson_cosine`: 0.8509553350510896
- `spearman_cosine`: 0.8484676559567688
- `pearson_manhattan`: 0.8503896832470704
- `spearman_manhattan`: 0.8475760325664419
- `pearson_euclidean`: 0.8513115442079158
- `spearman_euclidean`: 0.8484676559567688
- `pearson_dot`: 0.8489553386816947
- `spearman_dot`: 0.8484676559567688
- `pearson_max`: 0.8513115442079158
- `spearman_max`: 0.8484676559567688

These scores indicate the performance of the model in terms of cosine similarity across different metrics, where higher values suggest better alignment between the embeddings.

x??

---

