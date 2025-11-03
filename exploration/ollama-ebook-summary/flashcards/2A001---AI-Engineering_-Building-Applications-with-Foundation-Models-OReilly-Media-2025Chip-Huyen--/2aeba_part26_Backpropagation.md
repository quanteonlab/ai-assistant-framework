# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 26)

**Starting Chapter:** Backpropagation and Trainable Parameters

---

#### Memory Bottleneck Overview
Memory bottlenecks are a significant challenge when working with large models, particularly during finetuning. This is because finetuning requires more memory than inference due to the additional computations involved.

:p What are the primary factors contributing to a model's memory footprint during finetuning?
??x
The key contributors to a model’s memory footprint during finetuning are its number of parameters, trainable parameters, and numerical representations. 
??x
The answer includes that the more trainable parameters a model has, the higher its memory footprint will be. Reducing the number of trainable parameters is a strategy used in parameter-efficient finetuning (PEFT).

---

#### Backpropagation Explanation
Backpropagation is a fundamental mechanism for training neural networks, especially during finetuning. It involves two main phases: forward pass and backward pass.

:p What are the two main phases involved in backpropagation?
??x
The two main phases involved in backpropagation are the forward pass and the backward pass.
??x
The answer explains that these phases are crucial for updating model weights based on computed outputs compared to expected outputs. The code example here is a simplified illustration.

```java
public class BackpropExample {
    private double[] input;
    private double[] output;
    
    public void forwardPass() {
        // Compute the output using the current parameters and activation functions
        this.output = computeOutput(this.input);
    }
    
    private double[] computeOutput(double[] input) {
        // Simplified computation logic
        return new double[input.length];
    }
}
```
x??

---

#### Trainable Parameters vs. Frozen Parameters
Trainable parameters are those that can be updated during finetuning, while frozen parameters remain unchanged.

:p What distinguishes trainable parameters from frozen parameters?
??x
Trainable parameters are parameters that can be updated during finetuning, whereas frozen parameters are kept unchanged and not updated.
??x
The answer clarifies the distinction between trainable and frozen parameters. Frozen parameters do not need to compute their gradients, reducing memory requirements.

---

#### Quantization Techniques
Quantization is a method for converting models from higher bit formats (e.g., FP32) to lower bit formats (e.g., INT8), thereby reducing memory footprint.

:p What does quantization refer to in the context of model training and finetuning?
??x
Quantization refers to the practice of converting a model from a format with more bits to a format with fewer bits.
??x
The answer explains that this technique is used to reduce the memory requirements for both inference and finetuning. For example, a 13 billion parameter model using FP32 would require 52 GB, but if each value can be reduced to 2 bytes, the required memory drops to 26 GB.

---

#### Training Precision
Training typically requires higher numerical precision than inference. Mixed precision training is commonly used in deep learning models, where some operations are done in lower precision (e.g., 16-bit) and others in higher precision (e.g., 32-bit).

:p What is mixed precision training?
??x
Mixed precision training involves using different levels of numerical precision during the training process. Some operations use higher precision (e.g., 32-bit), while others use lower precision (e.g., 16-bit or 8-bit).
??x
The answer explains that this approach balances computational efficiency and model accuracy, allowing for reduced memory usage without significantly compromising performance.

---

#### Optimizer Overview
Optimizers are used to adjust the values of trainable parameters based on their gradients. Common optimizers include SGD (Stochastic Gradient Descent) and Adam, with Adam being widely used in transformer-based models.

:p What is an optimizer in the context of training neural networks?
??x
An optimizer is a method for adjusting the values of trainable parameters during training based on their gradients.
??x
The answer explains that optimizers like SGD or Adam help determine how much each parameter should be adjusted given its gradient value. An example code snippet can illustrate this.

```java
public class OptimizerExample {
    private double learningRate;
    
    public void updateParameters(double[] parameters, double[] gradients) {
        // Update the parameters using the optimizer's logic
        for (int i = 0; i < parameters.length; i++) {
            parameters[i] -= this.learningRate * gradients[i];
        }
    }
}
```
x??

---

#### Inference Memory Calculation for Models

Background context: During inference, only the forward pass is executed. The memory required includes model weights and activation values. Transformer models also need additional memory for attention mechanism's key-value vectors.

Relevant formulas:
- Model parameter count \( N \)
- Memory per parameter \( M \)

Memory needed to load the model’s parameters: 
\[ \text{Total Memory} = N \times M \]

For many applications, activation values and key-value vectors are 20% of the memory for the model's weights. Therefore:
\[ \text{Total Inference Memory} = N \times M \times 1.2 \]

:p What is the total inference memory needed for a 13B-parameter model with each parameter requiring 2 bytes?
??x
The total inference memory needed would be calculated as follows:

Given:
- \( N = 13,000,000,000 \) parameters (13B)
- \( M = 2 \) bytes per parameter

Total memory for model’s weights: 
\[ 13,000,000,000 \times 2 \text{ bytes} = 26 \text{ GB} \]

Since the activation values and key-value vectors are assumed to be 20% of the model's weights:
\[ \text{Total Inference Memory} = 26 \text{ GB} \times 1.2 = 31.2 \text{ GB} \]
x??

---

#### Training Memory Calculation for Models

Background context: During training, you need memory for model’s weights and activations (as discussed in inference), gradients, and optimizer states.

Relevant formulas:
- Model parameter count \( N \)
- Memory per parameter \( M \)

Training memory is calculated as:
\[ \text{Training Memory} = \text{Model Weights} + \text{Activations} + \text{Gradients} + \text{Optimizer States} \]

:p What is the total training memory needed for a 13B-parameter model with each parameter requiring 2 bytes?
??x
For a 13B-parameter model, each parameter requires 2 bytes. Let's calculate the memory required:

Given:
- \( N = 13,000,000,000 \) parameters (13B)
- \( M = 2 \) bytes per parameter

Memory for model’s weights: 
\[ 13,000,000,000 \times 2 \text{ bytes} = 26 \text{ GB} \]

If we assume the memory needed for activations is less than that of the model's weights, but considering practical scenarios where activation memory can be much larger:
\[ \text{Total Training Memory} = \text{Model Weights} + \text{Activations} + \text{Gradients} + \text{Optimizer States} \]

For simplicity, let’s assume activations are 20% of the model's weights:
\[ \text{Activation Memory} = 26 \text{ GB} \times 0.2 = 5.2 \text{ GB} \]

Now, using Adam optimizer (which stores two values per parameter for gradients and optimizer states):
\[ \text{Gradients + Optimizer States} = 13,000,000,000 \times 3 \times 2 \text{ bytes} / 8 = 96 \text{ GB} \]

Therefore:
\[ \text{Total Training Memory} = 26 \text{ GB} + 5.2 \text{ GB} + 96 \text{ GB} = 127.2 \text{ GB} \]
x??

---

#### Gradient Checkpointing and Activation Recomputation

Background context: To reduce memory requirements, one can use techniques like gradient checkpointing (also known as activation recomputation) where activations are not stored but recomputed when necessary.

:p What is gradient checkpointing, and why might it be used?
??x
Gradient checkpointing, also known as activation recomputation, is a technique to reduce the memory needed for storing activations during training. Instead of keeping all intermediate activation values in memory, these values can be computed on-the-fly whenever they are needed for backpropagation.

While this reduces memory requirements, it increases computational time because some operations need to be performed twice (once to generate the activations and once to compute the gradients).

:x??

---

#### Floating Point Formats Overview
Background context: The text discusses various floating point formats used in neural networks, including FP32, FP64, and FP16. These formats are essential for understanding how numerical values are represented and stored in models.

:p What are the common floating point formats discussed in the text?
??x
The common floating point formats discussed are:
- FP32 (Single precision): Uses 32 bits.
- FP64 (Double precision): Uses 64 bits.
- FP16 (Half precision): Uses 16 bits.

These formats differ primarily in their memory usage and precision capabilities. 
??x
---

#### BFloat16 Format
Background context: The text mentions that Google introduced BFloat16 as a format optimized for Cloud TPUs, emphasizing its efficiency while maintaining compatibility with FP32.

:p What is the BFloat16 format?
??x
BFloat16 (BFloat) uses 16 bits and is designed to be more memory-efficient than FP32 but retains FP32's precision. It is specifically optimized for AI performance on TPUs.
??x
---

#### Integer Formats Overview
Background context: The text introduces integer formats as an alternative to floating point formats, with INT8 (8-bit integers) and INT4 (4-bit integers) being common.

:p What are the common integer formats mentioned?
??x
The common integer formats mentioned are:
- INT8 (8-bit integers)
- INT4 (4-bit integers)

These formats represent numbers using a fixed number of bits, typically for reduced memory usage.
??x
---

#### Sign Bit and Mantissa Explanation
Background context: The text explains that each float format has 1 bit to represent the sign of the number. The rest of the bits are split between range (exponent) and precision (significand).

:p What does a floating point format consist of?
??x
A floating point format consists of:
- Sign Bit: Determines whether the number is positive or negative.
- Exponent: Determines the range of values that can be represented.
- Mantissa (Significand): Determines the precision with which numbers are represented.

For example, in a 32-bit FP32 format, there is 1 bit for the sign, 8 bits for the exponent, and 23 bits for the mantissa. 
??x
---

#### Range and Precision Bits
Background context: The text explains how range and precision bits affect the representable values and their accuracy.

:p What do "range" and "precision" refer to in floating point formats?
??x
- **Range**: Refers to the minimum and maximum values a format can represent. More exponent bits allow for a wider range.
- **Precision**: Refers to how accurately numbers are represented within that range. Fewer mantissa bits reduce precision.

For example, reducing FP32's 23-bit mantissa to FP16's 10-bit mantissa decreases the accuracy of representing fractional parts.
??x
---

#### Conversion Between Formats
Background context: The text provides examples of converting between different floating point formats and discusses potential inaccuracies due to precision loss.

:p How does converting from a higher-precision format (e.g., FP32) to a lower-precision format (e.g., FP16, BF16, TF32) affect the value?
??x
Converting from a higher-precision format like FP32 to lower formats like FP16 or BF16 can lead to inaccuracies due to reduced precision. For example:
- 0.0123456789 in FP32 becomes 0.01234 in FP16 and 0.01234 in BF16, losing some decimal precision.
- Values like 1234.56789 become rounded to the nearest representable value, such as 1235.0 or 1232.0, causing a slight change.

These conversions can result in minor changes to values and potential errors if high precision is required.
??x
---

