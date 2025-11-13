# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 36)

**Starting Chapter:** Chapter 12. Fine-Tuning Generation Models. The Three LLM Training Steps Pretraining Supervised Fine-Tuning and Preference Tuning

---

#### Language Modeling
Background context: The first step in creating a high-quality Large Language Model (LLM) is to pretrain it on one or more massive text datasets. During this training, the model attempts to predict the next token based on an input without labels, aiming to learn linguistic and semantic representations.

:p What is language modeling?
??x
Language modeling is the process of training a neural network model to predict the probability distribution over words in a sentence given its context. It helps the model learn patterns from large amounts of text data.
```python
# Pseudocode for simple language model training loop
for epoch in range(num_epochs):
    for batch in dataset:
        input_ids = batch['input_ids']
        labels = batch['labels']  # Usually same as input_ids, shifted by one token
        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
x??

---

#### Supervised Fine-Tuning (SFT)
Background context: After pretraining the base model, supervised fine-tuning is used to adapt it to follow instructions better. This involves training on labeled data where each input has an associated target output.

:p What is supervised fine-tuning?
??x
Supervised fine-tuning is a process of adapting a pretrained language model to specific tasks by fine-tuning its parameters using labeled data, often in the form of instruction-following or specific task completion.
```python
# Pseudocode for SFT training loop
for epoch in range(num_epochs):
    for batch in dataset:
        input_ids = batch['input_ids']
        labels = batch['labels']  # User inputs with desired outputs
        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
x??

---

#### Preference Tuning
Background context: The final step in enhancing an LLM is preference tuning. This process aligns the model's output with human preferences or AI safety standards by fine-tuning on data that reflects these preferences.

:p What is preference tuning?
??x
Preference tuning is a method of fine-tuning an LLM to better align its outputs with specific preferences or safety guidelines defined through additional training data. It involves adjusting the model parameters based on user-defined preferences.
```python
# Pseudocode for preference tuning loop
for epoch in range(num_epochs):
    for batch in dataset:
        input_ids = batch['input_ids']
        labels = batch['labels']  # Preference scores or ratings given by users
        loss = model(input_ids=input_ids, labels=labels).loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```
x??

---

#### The Three LLM Training Steps: Pretraining, Supervised Fine-Tuning, and Preference Tuning
Background context: The process of creating a high-quality language model involves three main steps: pretraining to learn general language patterns, supervised fine-tuning for task-specific instruction following, and preference tuning to align outputs with human preferences or safety standards.

:p What are the three main training steps for an LLM?
??x
The three main training steps for an LLM are:
1. **Pretraining**: Training on large datasets to learn general language patterns.
2. **Supervised Fine-Tuning (SFT)**: Adapting the model to follow specific instructions by fine-tuning with labeled data.
3. **Preference Tuning**: Aligning the model's outputs with human preferences or safety standards through additional training.

Example flow:
```python
# Example of a combined workflow
pretrained_model = train_language_model(dataset)
sft_model = fine_tune_sft(pretrained_model, sft_dataset)
tuned_model = fine_tune_preference(sft_model, preference_dataset)
```
x??

---

#### Supervised Fine-Tuning (SFT)
Background context explaining that during pretraining, a model learns to predict the next word(s) in a sequence. However, this doesn't necessarily mean it will follow instructions given by users.

:p What is supervised fine-tuning (SFT)?
??x
Supervised Fine-Tuning (SFT) involves adapting a pre-trained language model to specific tasks using labeled data. While pretraining focuses on learning general language patterns, SFT allows the model to understand and execute more complex instructions provided in a structured format.

This process typically involves providing input-output pairs where the model is trained to map inputs (instructions) to outputs (responses). This helps the model learn task-specific behaviors.
x??

---

#### Pretrained Model Behavior
Background context explaining that pretraining focuses on predicting next words, whereas fine-tuning adapts the model for specific tasks.

:p How does a base or pretrained LLM behave during pretraining?
??x
During pretraining, a language model is trained to predict the next word(s) in a sequence. This process helps it learn general language patterns and understand basic linguistic structures. However, since no instructions are provided, the model will often attempt to complete questions rather than follow them.

For example:
```python
input: "What is the capital of France? "
output: "Paris"
```
The model tries to predict the next word(s) without understanding that it needs to provide a direct answer.
x??

---

#### Full Fine-Tuning Process
Background context explaining full fine-tuning involves updating all parameters using a smaller but labeled dataset. It's used for specific tasks like following instructions.

:p What is full fine-tuning?
??x
Full fine-tuning is the process of adapting a pre-trained model to perform specific tasks by updating all its parameters based on a smaller, labeled dataset. Unlike pretraining, where the model learns general language patterns without any task-specific instructions, full fine-tuning trains the model to follow given instructions.

For example:
```python
input: "What is the capital of France? "
output: "Paris"
```
The model is trained to produce a direct answer based on user input.
x??

---

#### Example Full Fine-Tuning Data
Background context explaining that full fine-tuning uses labeled data, such as queries and corresponding answers.

:p What kind of data can be used for full fine-tuning?
??x
For full fine-tuning, any labeled data where inputs (queries) have corresponding outputs (answers) can be used. This allows the model to learn task-specific behaviors and improve its performance on specific tasks.

Example dataset:
```plaintext
input: "What is the capital of France? "
output: "Paris"

input: "Translate 'hello' into French"
output: "bonjour"
```
x??

---

#### Model Behavior After Fine-Tuning
Background context explaining that after fine-tuning, the model can follow instructions and produce relevant outputs.

:p How does a fine-tuned LLM behave differently from an un-fine-tuned one?
??x
After full fine-tuning, the language model is capable of following specific instructions provided in user queries. Instead of predicting next words or creating new questions, it generates appropriate responses based on the task at hand.

Example:
```python
input: "What is the capital of France? "
output: "Paris"
```
The fine-tuned model now understands that it needs to provide a direct answer rather than completing the question.
x??

---

---
#### Adapters for Parameter-Efficient Fine-Tuning (PEFT)
Adapters are a core component of many PEFT-based techniques. They introduce additional modular components inside the Transformer architecture to improve model performance on specific tasks without fully fine-tuning all the model weights.

Background context: 
- Adapters save time and compute resources by only fine-tuning 3.6% of BERT's parameters, which can achieve comparable or near-comparable performance to full fine-tuning.
- In the GLUE benchmark, the authors achieved results within 0.4% of full fine-tuning.

:p What are adapters in the context of parameter-efficient fine-tuning?
??x
Adapters are additional modular components inside a Transformer model that can be fine-tuned independently to enhance performance on specific tasks without fully training all the model's parameters. This approach saves significant time and computational resources.
x??

---
#### Architecture Placement of Adapters
In the proposed architecture, adapters are placed after both the attention layer and the feedforward neural network within each Transformer block.

:p Where are adapters placed in the Transformer block according to the paper?
??x
Adapters are placed after both the attention layer and the feedforward neural network within each Transformer block. This placement allows for efficient fine-tuning of a small subset of parameters while leaving most of the model's weights unchanged.
```python
# Pseudocode illustrating adapter placement in a single Transformer block
class AdapterBlock:
    def __init__(self, attention_layer, feed_forward_network):
        self.attention = attention_layer
        self.ffn = feed_forward_network
        self.adapter1 = Adapter()
        self.adapter2 = Adapter()

    def forward(self, x):
        # Process through the network
        x = self.attention(x)
        x = self.ffn(x)
        # Apply adapters after both layers
        x += self.adapter1(x)
        x += self.adapter2(x)
        return x
```
x??

---
#### Adapter Components Across Multiple Blocks
Adapter components span across multiple Transformer blocks, allowing for specialization in different tasks.

:p How do adapter components distribute across the model?
??x
Adapter components are distributed across all Transformer blocks within the model. This distribution allows for specialization of individual adapters to specific tasks. For example, one adapter can specialize in medical text classification while another can focus on named-entity recognition (NER).
```java
// Example class representing a single adapter component
public class Adapter {
    public void fineTune(double[] parameters) {
        // Fine-tuning logic here
    }
}

// Adapter components across multiple blocks
class Model {
    List<Adapter> adapters = new ArrayList<>();
    
    public void addAdaptersForTask(String task, double[] initialParams) {
        for (int i = 0; i < numBlocks; i++) {
            Adapter adapter = new Adapter();
            if ("medical".equals(task)) {
                adapter.fineTune(initialParams);
            } else if ("ner".equals(task)) {
                // Different fine-tuning logic
            }
            adapters.add(adapter);
        }
    }
}
```
x??

---
#### AdapterHub Framework
AdapterHub is a central repository for sharing adapters, primarily focused on BERT architectures initially but now applied to text generation Transformers.

:p What is AdapterHub?
??x
AdapterHub is a framework that serves as a central repository for sharing adapters. It has been used predominantly with BERT architectures but has recently been extended to text generation Transformers.
```java
// Example of downloading and using an adapter from AdapterHub
public class AdapterDownload {
    public void downloadAdapter(String task, String modelArchitecture) throws IOException {
        URL url = new URL("https://adapterhub.ml/tasks/" + task + "/" + modelArchitecture);
        HttpURLConnection connection = (HttpURLConnection) url.openConnection();
        InputStream responseStream = connection.getInputStream();
        
        // Process the downloaded adapter
    }
}
```
x??

---
#### Low-Rank Adaptation (LoRA)
Low-rank adaptation (LoRA) is an alternative technique to adapters that only requires updating a small set of parameters.

:p What is LoRA?
??x
Low-rank adaptation (LoRA) is a parameter-efficient fine-tuning technique that updates only a small subset of the model's parameters. It is designed as an alternative to adapter-based approaches and is known for its efficiency in fine-tuning large language models.
```python
# Pseudocode illustrating LoRA implementation
class LowRankAdapter:
    def __init__(self, rank):
        self.rank = rank
    
    def update_parameters(self, input_features, target_output):
        # Update parameters based on the low-rank approximation
        pass

# Example usage of LoRA in a model
class ModelWithLoRA(Model):
    def __init__(self):
        super().__init__()
        self.low_rank_adapter = LowRankAdapter(rank=2)
    
    def forward(self, x):
        # Normal forward pass
        y = super().forward(x)
        
        # Apply low-rank adaptation
        updated_params = self.low_rank_adapter.update_parameters(x, y)
        return y + updated_params
```
x??

---

#### Low-Rank Adaptation (LoRA)
Background context explaining LoRA. It is a method for fine-tuning large language models (LLMs) by creating a small subset of parameters instead of adding layers to the model, which can be kept separately from the base LLM. This approach allows for much quicker fine-tuning as only a small part of the base model needs to be updated.
:p What is LoRA and how does it differ from traditional fine-tuning methods?
??x
LoRA is a technique for fine-tuning large language models (LLMs) by creating a smaller subset of parameters that can replace parts of the original LLM. Unlike traditional fine-tuning, which involves updating all layers of the model, LoRA updates only a small portion of the base model's parameters.

The key difference lies in efficiency and resource usage: while full fine-tuning requires updating millions or billions of parameters (depending on the size of the LLM), LoRA reduces this by approximating large matrices with smaller ones. This makes it much faster to train and less computationally intensive.
x??

---

#### Approximation of Weight Matrices
Background context explaining how weight matrices in large language models can be approximated using smaller matrices, leading to significant efficiency gains during training.
:p How does LoRA approximate the weight matrices used in LLMs?
??x
LoRA approximates the weight matrices found in large language models by breaking them into smaller, lower-rank matrices. This is achieved through matrix decomposition techniques where a single large weight matrix $W $ of size$M \times N $ can be represented as a product of two smaller matrices:$U \cdot V^T $, where$ U $ and $ V $ are much smaller in size than $ W$.

For example, if we have a 10x10 weight matrix:
$$W = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1,10}\\
w_{21} & w_{22} & \cdots & w_{2,10}\\
\vdots & \vdots & \ddots & \vdots\\
w_{10,1} & w_{10,2} & \cdots & w_{10,10}
\end{bmatrix}$$

LoRA can approximate this matrix using two smaller matrices:
$$

W \approx U \cdot V^T = \begin{bmatrix}
u_1 \\
u_2 \\
\vdots \\
u_{10}
\end{bmatrix} \cdot
\begin{bmatrix}
v_1 & v_2 & \cdots & v_{10}
\end{bmatrix}$$

This reduces the number of parameters from 100 (for a full matrix) to just 20, significantly reducing computational load.
x??

---

#### Efficiency in Fine-Tuning
Background context on why efficiency is crucial for fine-tuning large language models and how LoRA addresses this issue by using smaller matrices.
:p Why is efficiency important in the fine-tuning of large language models?
??x
Efficiency is crucial in fine-tuning large language models because these models often contain billions or even trillions of parameters. Training such a model requires substantial computational resources, making the process slow and resource-intensive.

LoRA addresses this issue by reducing the number of parameters that need to be updated during training. Instead of fine-tuning all layers, LoRA focuses on updating only a small subset of parameters by approximating large matrices with smaller ones. This reduces both the time required for training and the computational resources needed, making it more feasible to perform fine-tuning.

For example, consider an LLM with 175 billion parameters in each Transformer block. If we can reduce the rank from 12,288x12,288 to a lower rank like 8, this would require only two smaller matrices of size 12,288x2, significantly reducing the number of parameters to be updated.
x??

---

#### Flexibility in Fine-Tuning
Background context on how LoRA allows for selective fine-tuning of specific parts of the base model and provides flexibility in training.
:p How does LoRA provide flexibility in fine-tuning large language models?
??x
LoRA provides flexibility by allowing users to selectively fine-tune specific parts of the base model. Unlike traditional full fine-tuning, which updates all parameters, LoRA enables updating only a small subset of the model's parameters.

For instance, during training, you might choose to fine-tune only certain weight matrices in each Transformer layer, such as the Query and Value matrices. This selective approach ensures that the model can be adapted without overwhelming computational resources, making it more practical for various applications.

By focusing on specific parts of the model, LoRA achieves a balance between performance and resource efficiency, allowing for more targeted improvements where needed.
x??

---

#### Intrinsic Dimensionality
Background context explaining how intrinsic dimensionality affects the effectiveness of language model fine-tuning, as demonstrated by research.
:p How does intrinsic dimensionality explain the effectiveness of language model fine-tuning?
??x
Intrinsic dimensionality is a concept that describes the effective number of parameters required to capture the essential information in a high-dimensional space. Research has shown that large language models have a very low intrinsic dimension, meaning they can be well-approximated using much smaller matrices.

This insight is crucial for methods like LoRA, which aims to update only a small subset of the model's parameters during fine-tuning. Since the effective number of parameters required is lower than the full matrix size, approximating these with smaller matrices leads to significant efficiency gains without compromising performance.

For example, in a 175 billion parameter model like GPT-3, each Transformer block contains a weight matrix of 12,288x12,288. By reducing this to a rank 8 approximation (using two smaller matrices), the number of parameters needed is drastically reduced, leading to faster and more efficient training.
x??

---

#### Decomposition for Training Efficiency
Background context on how decomposing large weight matrices into smaller matrices helps in making fine-tuning more efficient during training.
:p How does decomposing large weight matrices help in making fine-tuning more efficient?
??x
Decomposing large weight matrices into smaller, lower-rank matrices helps in making the fine-tuning process more efficient. This is achieved by approximating a large matrix $W $ of size$M \times N $ using two smaller matrices:$U \cdot V^T $, where$ U $ and $ V$ are much smaller.

By doing this, the number of parameters required for training is significantly reduced. For example, if we have a 10x10 matrix:
$$W = \begin{bmatrix}
w_{11} & w_{12} & \cdots & w_{1,10}\\
w_{21} & w_{22} & \cdots & w_{2,10}\\
\vdots & \vdots & \ddots & \vdots\\
w_{10,1} & w_{10,2} & \cdots & w_{10,10}
\end{bmatrix}$$

We can approximate this matrix using two smaller matrices:
$$

W \approx U \cdot V^T = \begin{bmatrix}
u_1 \\
u_2 \\
\vdots \\
u_{10}
\end{bmatrix} \cdot
\begin{bmatrix}
v_1 & v_2 & \cdots & v_{10}
\end{bmatrix}$$

This reduces the number of parameters from 100 (for a full matrix) to just 20, making the training process much faster and more resource-efficient.

During training, only these smaller matrices are updated, while the full weight changes are combined with the frozen base weights.
x??

---

#### Weight Representation and Precision
Background context explaining how weights in large language models (LLMs) are represented with a specific precision, typically float64 or float32. Reducing the number of bits can decrease memory requirements but also reduces accuracy as shown in Figure 12-15.

:p How does reducing the number of bits affect weight representation and model memory?
??x
Reducing the number of bits used to represent weights decreases the precision, which can lead to less accurate results. However, it also lowers the memory requirements for storing these weights, making the model more efficient. As illustrated in Figure 12-15, using a float32 representation compared to a float16 representation leads to a loss in accuracy.

```java
// Example of weight initialization with different precisions
float[] float64Weights = new float[1000]; // High precision, high memory
float[] float32Weights = new float[1000]; // Lower precision, lower memory
```
x??

---

#### Quantization and Its Challenges
Background context explaining the need for quantization to reduce model size while maintaining accuracy. Directly mapping higher precision values to lower precision can result in loss of differentiating factors as shown in Figure 12-16.

:p What are the challenges when directly quantizing weights from high to low precision?
??x
Directly mapping higher precision values to lower precision can cause multiple original weight values to be represented by the same lower precision value, leading to a loss of differentiation and accuracy. For example, if we map 32-bit values to 16-bit values, some close but distinct weights might end up being quantized to the same value.

```java
// Pseudocode for direct mapping challenge
float[] highPrecisionValues = [0.999f, 1.001f]; // Example high precision values
short[] lowPrecisionValues; // These could map to the same value if directly mapped

for (int i = 0; i < highPrecisionValues.length; i++) {
    int quantizedValue = (int) ((highPrecisionValues[i] + 1.0f) * 32768); // Direct mapping
    lowPrecisionValues[i] = (short) quantizedValue;
}
```
x??

---

#### Blockwise Quantization Method
Background context explaining how blockwise quantization, introduced by QLoRA, allows for efficient and accurate quantization of weights. This method uses additional blocks to prevent the loss of differentiating factors between close values.

:p What is blockwise quantization, and why is it useful?
??x
Blockwise quantization is a technique that enhances quantization by creating additional blocks or sub-blocks to represent similar weights accurately with lower precision. It helps in maintaining differentiation between closely valued weights without reducing accuracy too much. This approach prevents the loss of differentiating factors that can occur when directly mapping higher precision values to lower precision, as shown in Figure 12-16.

```java
// Pseudocode for blockwise quantization
class BlockQuantizer {
    Map<Short, List<Float>> blockMap = new HashMap<>();

    public void addBlock(float value) {
        int blockIndex = findClosestBlock(value); // Find the closest existing block or create a new one
        blockMap.computeIfAbsent(blockIndex, k -> new ArrayList<>()).add(value);
    }

    private int findClosestBlock(float value) {
        // Logic to find the closest block index based on value
        return 0;
    }
}
```
x??

---

#### Distribution-Aware Quantization
Background context explaining that neural network weights are generally normally distributed between -1 and 1. This distribution can be leveraged for more efficient quantization as shown in Figure 12-18.

:p How does the normal distribution of weights impact quantization?
??x
The normal distribution of weights in neural networks, typically centered around zero with values ranging from -1 to 1, allows for more efficient quantization. By understanding this distribution, we can bin or group similar weight values together and quantize them accurately using fewer bits. This approach helps mitigate issues with outliers that might otherwise cause inaccuracies.

```java
// Pseudocode for distribution-aware quantization
public class DistributionAwareQuantizer {
    private Map<Double, List<Float>> bins = new HashMap<>();

    public void quantize(float value) {
        double normalizedValue = (value + 1.0f) / 2; // Normalize to [0, 1]
        int binIndex = findClosestBin(normalizedValue); // Find the closest bin index
        List<Float> valuesInBin = bins.computeIfAbsent(binIndex, k -> new ArrayList<>());
        valuesInBin.add(value);
    }

    private int findClosestBin(double value) {
        // Logic to find the closest bin based on normalized value
        return 0;
    }
}
```
x??

---

