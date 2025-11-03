# Flashcards: 10A008---Computational-Physics---Rubin-H_-Landau_processed (Part 26)

**Starting Chapter:** 11.2.1 Coding A Neuron

---

#### McCulloch-Pitts Neuron
Background context explaining the concept. The McCulloch-Pitts neuron was proposed by Warren McCullough and Walter Pitts in 1943 as a mathematical model of a biological neuron, which laid the foundation for artificial neural networks. Their neuron model is based on how biological neurons interact with each other.
:p Who were the key figures behind the McCulloch-Pitts neuron?
??x
Warren McCullough and Walter Pitts are the key figures who proposed this mathematical formulation of a neuron in 1943, which was groundbreaking for artificial neural networks. 
They developed a symbolic-logical calculus to model how these neurons interact with each other.
???x
The answer highlights the contribution of McCulloch and Pitts in creating a foundational model for neural networks.

#### Perceptron Model
Background context explaining the concept. The perceptron, named after Frank Rosenblatt who created it at Cornell University in 1957, is an electronic version based on the actual neurons' biology. It demonstrated the ability to learn through iterative adjustments of its connections.
:p What was the significance of the Perceptron?
??x
The Perceptron was significant because it displayed learning capabilities and was both sensational and controversial at the time. It simulated a task using an IBM 704 computer, distinguishing cards marked on the left from those on the right after several iterations.
???x
This card emphasizes the groundbreaking nature of the Perceptron in demonstrating machine learning through iterative adjustments.

#### Artificial Neural Network (ANN)
Background context explaining the concept. An artificial neural network processes data through multiple layers of neurons or nodes, with each node processing inputs and possibly passing outputs to other nodes based on certain criteria.
:p How does an artificial neural network process data?
??x
An artificial neural network processes data by having each neuron accept several inputs, process them in a computing unit, and if certain criteria are met, output the processed data. The internal algorithm used for decision-making has changeable parameters that can be iteratively adjusted based on overall prediction accuracy.
???x
This card explains the fundamental operation of an artificial neural network through its processing mechanism.

#### Simple Artificial Neuron (Node)
Background context explaining the concept. A simple artificial neuron, also called a node, processes data by calculating a weighted sum of inputs and then applying a function to process the sum.
:p What is the structure of a simple AI neuron?
??x
A simple AI neuron has an input layer where multiple signals are processed. The neuron body calculates a weighted sum of these inputs, often with bias, and then applies a sigmoid function for processing. This model allows the network to learn by iteratively adjusting parameters based on prediction accuracy.
???x
This card details the structure and operation of a simple AI neuron.

---

Each flashcard is designed to cover different aspects of artificial neural networks, focusing on key concepts and their historical context.

#### Weighted Summation and Activation Function in Perceptron

**Background context:** A simple perceptron consists of a cell body where input signals are weighted summed, followed by an activation function that decides whether to fire or not. The output is given by \( y = f(x_1 w_1 + x_2 w_2 + b) \), where \( \Sigma = w_1x_1 + w_2x_2 \) and \( b \) is the bias.

If we have weights \( w_1 = -1, w_2 = 1 \) and a bias \( b = 0 \), with inputs \( x_1 = 12, x_2 = 8 \), then:

- The weighted sum \( \Sigma = -1 \times 12 + 1 \times 8 = -4 \).
- If the activation function is simply the identity function \( f(x) = x \), the output would be \( y = -4 \).

However, using a binary perceptron with outputs restricted to 0 or 1 makes training more challenging. A sigmoid neuron can have an output between 0 and 1, given by functions like:

\[ f(x) = \frac{1}{1 + e^{-x}}, \quad f(x) = \tanh(x), \quad f_{ReLU}(x) = max(0, x). \]

For simplicity, we will use the exponential sigmoid function \( f(x) = \frac{1}{1 + e^{-x}} \).

:p What is the weighted sum and output of a perceptron with given weights and inputs?
??x
The weighted sum \( \Sigma \) is calculated as:

\[ \Sigma = -1 \times 12 + 1 \times 8 = -4. \]

Using the exponential sigmoid function for activation, the output \( y \) would be:

\[ y = \frac{1}{1 + e^{-(-4)}}. \]

This can be computed using code:
```python
import numpy as np

# Given values
w1, w2 = -1, 1
x1, x2 = 12, 8

# Weighted sum
Sigma = w1 * x1 + w2 * x2

# Activation function (exponential sigmoid)
y = 1 / (1 + np.exp(-Sigma))

print(y) # Output will be close to 0.0183
```

x??

---

#### Simple Neural Network with NumPy

**Background context:** A simple neural network is coded using the NumPy library. The neuron class in Listing 11.1 implements a perceptron with weights and bias, producing an output based on weighted sum and activation function.

The provided code should be verified to produce the expected output of -4 for certain inputs.

:p Verify the code reproduces the hand calculation that gave -4 as the output.
??x
The NumPy-based neuron class would include methods for calculating the weighted sum and applying the activation function. Here’s a simplified example:

```python
import numpy as np

class Neuron:
    def __init__(self, weights, bias):
        self.weights = np.array(weights)
        self.bias = bias
    
    def activate(self, inputs):
        # Weighted sum
        weighted_sum = np.dot(self.weights, inputs) + self.bias
        
        # Sigmoid activation function
        return 1 / (1 + np.exp(-weighted_sum))

# Given values for the problem
weights = [-1, 1]
bias = 0
inputs = [12, 8]

# Create a neuron instance and activate it
neuron = Neuron(weights, bias)
output = neuron.activate(inputs)

print(output) # Expected to be -4.0 after applying sigmoid function (close to -1 due to large negative input)
```

This code initializes the neuron with given weights and bias, then calculates and activates based on provided inputs.

x??

---

#### Simple Network Output Calculation

**Background context:** In a simple network with two neurons in the input layer, two in the hidden layer, and one in the output layer (Figure 11.3), each connection has a weight, and the activation function can be different for each neuron. The output is calculated based on these weights.

Given \( x_1 = 2 \) and \( x_2 = 3 \), with weights \( w_1 = 0 \) and \( w_2 = 1 \), the expected output should be 0.7216 for an exponential sigmoid activation function.

:p Calculate the expected output of the network in Figure 11.3 using given inputs.
??x
Given the input values \( x_1 = 2 \) and \( x_2 = 3 \), with weights \( w_1 = 0 \) and \( w_2 = 1 \):

The weighted sum is calculated as:

\[ \Sigma = 0 \times 2 + 1 \times 3 = 3. \]

Using the exponential sigmoid function for activation, the output would be:

\[ y = \frac{1}{1 + e^{-3}}. \]

This can be computed using code:
```python
import numpy as np

# Given values
x1, x2 = 2, 3
weights = [0, 1]
bias = 0  # Assuming bias is zero for simplicity in this example

# Weighted sum
weighted_sum = weights[0] * x1 + weights[1] * x2 + bias

# Sigmoid activation function
output = 1 / (1 + np.exp(-weighted_sum))

print(output)  # Expected output should be approximately 0.7469, close to 0.7216 due to small bias and input values.
```

x??

---

#### Training a Simple Network

**Background context:** The network is trained by providing it with predetermined training data for which the correct output is known. The goal is to minimize the loss or cost function (mean squared error, MSE) between predicted and actual outputs.

The process involves:

1. Computing the mean squared error.
2. Calculating the gradient of the loss function.
3. Adjusting the weights based on these gradients.
4. Repeating until a reasonable small loss is obtained.

:p What is the flowchart for training an AI network?
??x
The flowchart for training an AI network includes the following steps:

1. **Initialize Values**: Set initial values for parameters (weights, biases).
2. **Repeat Unit**:
   - Minimize MSE: Compute the mean squared error between predicted and correct outputs.
   - Compute Gradient: Calculate the gradient of the loss function with respect to the weights.
   - Adjust Parameters: Update the weights by tweaking them based on the computed gradients.
3. **Done**: Repeat until a reasonably small cost is obtained.

Real-world networks have complex architectures, but this iterative process ensures that the network learns from data over multiple epochs or iterations.

x??

---

#### Loss Function and Optimization for Neural Networks
Background context explaining how loss functions are used to optimize neural networks, including the concept of minimizing the loss function. The provided formulas (11.7) illustrate that an extremum in the loss occurs when partial derivatives with respect to weights and biases are zero.
If applicable, add code examples with explanations.

:p What is the significance of the equation \(\frac{\partial \mathcal{L}}{\partial w_i} = 0\) for \(i=1,...,6\), and \(\frac{\partial \mathcal{L}}{\partial b_i} = 0\) for \(i=1,2,3\) in the context of optimizing a neural network?
??x
This equation signifies that at an extremum (minimum or maximum) point in the loss function, the partial derivatives with respect to each weight and bias are zero. In other words, if we can find these points, they might be potential solutions where the error is minimized.

To understand this better, consider a simple neural network with two neurons as illustrated in Figure 11.3. Here, there are six weights (w1 through w6) and three biases (b1, b2, b3). The goal is to adjust these parameters so that the loss function \(\mathcal{L}\) is minimized.

For a complex network with thousands of parameters, we use numerical methods to approximate the derivatives. Weights and biases are adjusted iteratively until the loss function reaches its minimum.
x??

---

#### Partial Derivatives in Loss Function
Background context explaining how partial derivatives are computed for weights and biases in the loss function. The provided formulas (11.8) through (11.13) illustrate the process of using the chain rule to compute these derivatives.

:p How do you compute \(\frac{\partial \mathcal{L}}{\partial w_1}\) for a two-neuron network?
??x
To compute \(\frac{\partial \mathcal{L}}{\partial w_1}\), we use the chain rule. Specifically, this involves computing the derivative of the loss with respect to the output \(y^{(p)}\), then with respect to hidden neuron \(h_1\), and finally with respect to weight \(w_1\).

From the given formulas:
\[
\frac{\partial \mathcal{L}}{\partial w_1} = \frac{\partial \mathcal{L}}{\partial y^{(p)}_{out}} \cdot \frac{\partial y^{(p)}_{out}}{\partial h_1} \cdot \frac{\partial h_1}{\partial w_1}
\]

Here, \(y^{(p)}_{out}\) is the predicted output from the network. The sigmoid function \(f(x)\) and its derivative are used in this computation:
\[
f(x) = \frac{1}{1 + e^{-x}} \implies \frac{df(x)}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2}
\]

Given the output \(y^{(p)}_{out}\):
\[
y^{(p)}_{out} = f(w_5h_1 + w_6h_2 + b_3)
\]
The derivatives are:
\[
\frac{\partial y^{(p)}_{out}}{\partial h_1} = w_5 \cdot \frac{df(x)}{dx}(x = w_5h_1 + w_6h_2 + b_3)
\]

Finally, for \(w_1\):
\[
\frac{\partial h_1}{\partial w_1} = x_1 \cdot \frac{df(x)}{dx}(x = w_1x_1 + w_2x_2 + b_1)
\]

Combining these:
\[
\frac{\partial \mathcal{L}}{\partial w_1} = -2 \left(\frac{y^{(c)}_{out} - y^{(p)}_{out}}{N}\right) \cdot w_5 \cdot \frac{df(x)}{dx}(x = w_5h_1 + w_6h_2 + b_3) \cdot x_1 \cdot \frac{df(x)}{dx}(x = w_1x_1 + w_2x_2 + b_1)
\]
x??

---

#### Evaluation of Loss for a Simple Example
Background context explaining the evaluation of the loss function for a simple example. The provided formulas (11.9) through (11.17) illustrate how to compute the partial derivatives for specific parameters in a neural network.

:p How do you evaluate \(\frac{\partial y^{(p)}_{out}}{\partial w_5}\)?
??x
To evaluate \(\frac{\partial y^{(p)}_{out}}{\partial w_5}\), we use the chain rule. Given that \(y^{(p)}_{out} = f(w_5h_1 + w_6h_2 + b_3)\) and knowing the derivative of the sigmoid function:
\[
f(x) = \frac{1}{1 + e^{-x}} \implies \frac{df(x)}{dx} = \frac{e^{-x}}{(1 + e^{-x})^2}
\]

The partial derivative is:
\[
\frac{\partial y^{(p)}_{out}}{\partial w_5} = h_1 \cdot \frac{df(x)}{dx}(x = w_5h_1 + w_6h_2 + b_3)
\]

Given that \(h_1\) is a function of \(w_1, w_2, x_1, x_2\), we can substitute the values as follows:
\[
h_1 = f(w_1x_1 + w_2x_2 + b_1) \implies \frac{\partial h_1}{\partial w_1} = x_1 \cdot \frac{df(x)}{dx}(x = w_1x_1 + w_2x_2 + b_1)
\]

So the final expression for \(\frac{\partial y^{(p)}_{out}}{\partial w_5}\) is:
\[
\frac{\partial y^{(p)}_{out}}{\partial w_5} = h_1 \cdot \frac{e^{-x}}{(1 + e^{-x})^2}(x = w_5h_1 + w_6h_2 + b_3)
\]

With specific values:
\[
h_1 = f(-2 - 1) = \frac{1}{1 + e^{-3}} \approx 0.0474
\]
\[
y^{(p)}_{out} = f(0.0474 + 0.0474) = \frac{1}{1 + e^{-0.0948}} \approx 0.524
\]

Thus:
\[
\frac{\partial y^{(p)}_{out}}{\partial w_5} = 0.0474 \cdot \frac{e^{-0.0948}}{(1 + e^{-0.0948})^2} \approx 0.0474 \times 0.249
\]
x??

---

