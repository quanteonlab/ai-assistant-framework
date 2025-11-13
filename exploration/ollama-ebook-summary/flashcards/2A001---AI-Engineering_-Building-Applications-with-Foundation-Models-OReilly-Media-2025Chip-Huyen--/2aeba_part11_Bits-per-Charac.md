# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 11)

**Starting Chapter:** Bits-per-Character and Bits-per-Byte

---

#### Entropy
Background context explaining the concept of entropy and its relation to information. The formula for entropy is given as $H = -\sum p_i \log_2(p_i)$, where $ p_i$ represents the probability of a token.

:p What is entropy in the context of language models?
??x
Entropy measures, on average, how much information each token carries. A higher entropy value indicates that tokens carry more information but require more bits to represent them. This can be understood by considering a simple example where two different languages are used to describe positions within a square.
```java
public class EntropyExample {
    public static double calculateEntropy(double[] probabilities) {
        double entropy = 0;
        for (double prob : probabilities) {
            if (prob > 0) {
                entropy -= prob * Math.log2(prob);
            }
        }
        return entropy;
    }

    // Example usage
    public static void main(String[] args) {
        double[] probabilities = {0.5, 0.5};
        System.out.println("Entropy: " + calculateEntropy(probabilities));
    }
}
```
x??

---

#### Cross Entropy
Background context explaining the concept of cross entropy and its relevance to language models. The formula for cross entropy is given as $H(P||Q) = -\sum p_i \log_2(q_i)$, where $ P$represents the true distribution of training data, and $ Q$ is the distribution learned by the model.

:p What does cross entropy measure in a language model?
??x
Cross Entropy measures how difficult it is for the language model to predict what comes next in the dataset. It depends on two factors: 1) The predictability of the training data (measured by its entropy), and 2) How closely the distribution captured by the model matches the true distribution of the training data.
```java
public class CrossEntropyExample {
    public static double calculateCrossEntropy(double[] trueDistribution, double[] predictedDistribution) {
        double crossEntropy = 0;
        for (int i = 0; i < trueDistribution.length; i++) {
            if (trueDistribution[i] > 0 && predictedDistribution[i] > 0) {
                crossEntropy -= trueDistribution[i] * Math.log2(predictedDistribution[i]);
            }
        }
        return crossEntropy;
    }

    // Example usage
    public static void main(String[] args) {
        double[] trueDist = {0.5, 0.5};
        double[] predDist = {0.6, 0.4};
        System.out.println("Cross Entropy: " + calculateCrossEntropy(trueDist, predDist));
    }
}
```
x??

---

#### Perplexity
Background context explaining the concept of perplexity and its relation to cross entropy.

:p What is perplexity in language models?
??x
Perplexity is a measure derived from cross entropy. It gives an estimate of how well a probability distribution predicts a sample. Lower perplexity indicates better prediction accuracy, where 2^H(P||Q) = \exp(H(P||Q)) represents the perplexity.
```java
public class PerplexityExample {
    public static double calculatePerplexity(double crossEntropy) {
        return Math.exp(crossEntropy);
    }

    // Example usage
    public static void main(String[] args) {
        double crossEntropy = 1.0; // Assuming some value from previous calculations
        System.out.println("Perplexity: " + calculatePerplexity(crossEntropy));
    }
}
```
x??

---

#### Bits-Per-Character (BPC)
Background context explaining the concept of BPC and its relation to entropy and cross entropy.

:p What is bits-per-character in language models?
??x
Bits-Per-Character (BPC) measures how many bits are required on average to represent a character, based on the distribution learned by the model. It can be calculated as $\text{BPC} = -\frac{\log_2(P(x_{i+1}|x_i))}{n}$, where $ P(x_{i+1}|x_i)$is the probability of the next token given a context, and $ n$ is the number of tokens.
```java
public class BPCExample {
    public static double calculateBPC(double crossEntropy, int nTokens) {
        return -crossEntropy / Math.log(2) / nTokens;
    }

    // Example usage
    public static void main(String[] args) {
        double crossEntropy = 1.0; // Assuming some value from previous calculations
        int nTokens = 100; // Number of tokens in the dataset
        System.out.println("BPC: " + calculateBPC(crossEntropy, nTokens));
    }
}
```
x??

---

#### Bits-Per-Byte (BPB)
Background context explaining the concept of BPB and its relation to BPC.

:p What is bits-per-byte in language models?
??x
Bits-Per-Byte (BPB) measures how many bits are required on average to represent a byte, based on the distribution learned by the model. It can be calculated as $\text{BPB} = -\frac{\log_2(P(x_{i+1}|x_i))}{n/8}$, where $ P(x_{i+1}|x_i)$is the probability of the next token given a context, and $ n$ is the number of bytes in the dataset.
```java
public class BPBExample {
    public static double calculateBPB(double crossEntropy, int nBytes) {
        return -crossEntropy / Math.log(2) / (nBytes / 8);
    }

    // Example usage
    public static void main(String[] args) {
        double crossEntropy = 1.0; // Assuming some value from previous calculations
        int nBytes = 500; // Number of bytes in the dataset
        System.out.println("BPB: " + calculateBPB(crossEntropy, nBytes));
    }
}
```
x??

---

---
#### Cross Entropy and Kullback-Leibler Divergence
Cross entropy is a measure of the difference between two probability distributions. It quantifies how much one distribution (Q) deviates from another reference distribution (P). The formula for cross entropy $H(P, Q)$ with respect to P is given by:
$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

The Kullback-Leibler (KL) divergence measures the difference between these distributions in a way that is not symmetric. It quantifies how one probability distribution diverges from a second, expected probability distribution and is given by:
$$

D_{KL}(P || Q) = H(P, Q) - H(P)$$

The cross entropy isn't symmetric;$H(P, Q) \neq H(Q, P)$. The KL divergence of Q with respect to P is:
$$D_{KL}(P || Q) = \sum_{x} P(x) \log \left( \frac{P(x)}{Q(x)} \right)$$

A language model trained to minimize its cross entropy with respect to the training data will approximate the true distribution of the training data. If it learns perfectly, the cross entropy would match the entropy of the training data, and the KL divergence would be 0.

:p What is the formula for cross entropy $H(P, Q)$?
??x
The formula for cross entropy between two probability distributions P and Q is:
$$H(P, Q) = -\sum_{x} P(x) \log Q(x)$$

This quantifies how much distribution Q deviates from the reference distribution P.
x??

---
#### Cross Entropy vs. KL Divergence
Cross entropy measures the difficulty for a model to predict the next token in a sequence, while KL divergence measures the uncertainty or information needed to change one probability distribution into another.

:p How does cross entropy and KL divergence differ in their application?
??x
Cross entropy $H(P, Q)$ measures how well a model predicts the next token. It quantifies the expected number of bits required to encode messages generated by P using codes optimized for Q.

KL Divergence $D_{KL}(P || Q)$, on the other hand, measures the difference between two probability distributions. Specifically, it tells us how much information is needed to change distribution Q to match P.
x??

---
#### Bits-per-Character (BPC)
Bits per character (BPC) measures the efficiency of a language model in representing each token as bits. It helps compare models that use different tokenization methods.

For example, if a model has a cross entropy of 6 bits and each token on average consists of 2 characters:
$$\text{BPC} = \frac{\text{Cross Entropy (bits)}}{\text{Average Characters per Token}}$$

If BPC is 3 and ASCII encoding uses 7 bits, the bits-per-byte (BPB) can be calculated as:
$$\text{BPB} = \frac{\text{Bits per Character}}{\text{Bits per Character in Encoding Scheme}}$$:p What is the formula for calculating Bits-per-Character (BPC)?
??x
The formula for calculating Bits-per-Character (BPC) is:
$$\text{BPC} = \frac{\text{Cross Entropy (bits)}}{\text{Average Characters per Token}}$$

This metric helps in understanding how efficiently a language model can represent text.
x??

---
#### Perplexity
Perplexity measures the uncertainty or surprise of a model's predictions. It is derived from cross entropy and is defined as:
$$

PPL(P) = 2^{H(P)}$$

For a dataset with true distribution $P $ and learned distribution$Q$:
$$PPL(P, Q) = 2^{H(P, Q)}$$

Perplexity provides insight into the model's uncertainty when predicting the next token in a sequence.

:p What is the formula for Perplexity?
??x
The formula for Perplexity is:
$$\text{PPL} = 2^{\text{Cross Entropy}}$$

For a dataset with true distribution $P $ and learned distribution$Q$:
$$\text{PPL}(P, Q) = 2^{H(P, Q)}$$

Perplexity measures the uncertainty of predictions, making it useful in evaluating how well a model can predict sequences.
x??

---

#### Bit vs. Nat for Entropy and Cross Entropy
Background context explaining the concept. The text discusses the use of bits and nats as units for measuring entropy and cross entropy. Bits represent 2 unique values, while nats use the base $e$. Popular frameworks like TensorFlow and PyTorch use nats.
:p What are the differences between using bits and nats in measuring entropy and cross entropy?
??x
Bits and nats are used to measure entropy and cross entropy differently due to their bases. Bits are based on a binary system (base 2), which means each bit can represent two unique values, $0 $ or$1 $. In contrast, nats use the natural logarithm base $ e \approx 2.718$.

Using bits for these measurements involves calculations with base 2 logarithms, while using nats involves calculations with the natural logarithm (base $e$). For example:
- Entropy in bits: $H(X) = -\sum_{i} p(x_i) \log_2(p(x_i))$- Entropy in nats:$ H(X) = -\sum_{i} p(x_i) \ln(p(x_i))$ Consequently, perplexity calculations differ based on the unit used. When using bits, perplexity is given by:
$$PPL(P,Q) = 2^{H(P,Q)}$$

However, when using nats, perplexity becomes:
$$

PPL(P,Q) = e^{H(P,Q)}$$x??

---

#### Perplexity Interpretation and Use Cases
Background context explaining the concept. The text provides a detailed explanation of how perplexity is used to measure the uncertainty in predicting tokens by language models.
:p What does a higher or lower perplexity value indicate about a model's performance?
??x
A higher perplexity value indicates greater uncertainty in the model’s predictions, while a lower perplexity value indicates more accurate predictions. This means that with lower perplexity, the model can better predict the next token(s) in a sequence.

In terms of specific scenarios:
- **Structured Data**: More structured data tends to have lower expected perplexity because it is easier to predict. For example, HTML code has more predictable patterns compared to everyday text.
- **Vocabulary Size**: Larger vocabularies lead to higher perplexity due to the increased number of possible tokens. A model’s perplexity on a children's book (smaller vocabulary) would be lower than its perplexity on War and Peace (larger vocabulary).
- **Context Length**: Longer context lengths reduce uncertainty, leading to lower perplexity. Modern models can condition their predictions on up to 10,000 previous tokens or more.
x??

---

#### Examples of Perplexity in Different Contexts
Background context explaining the concept. The text provides examples comparing perplexity values for different types of data and contexts.
:p How does the perplexity value change with different types of data?
??x
Perplexity varies significantly based on the type of data being modeled:
- **More Structured Data**: HTML code is more predictable, leading to lower perplexity because the structure (like opening tags) helps in prediction. For instance, a model's perplexity might be 2 for predicting closing tags after an opening tag.
- **Vocabulary Size**: A smaller vocabulary means fewer possible tokens, making it easier to predict the next token accurately. Therefore, a model’s perplexity on a children's book would typically be lower than its perplexity on War and Peace due to the broader range of words in the latter.
- **Context Length**: With longer context lengths, models have more information to make accurate predictions, reducing uncertainty and lowering the perplexity value.

For example:
- Predicting the next character in a children's book might result in a perplexity of 3.
- Predicting the next word in War and Peace would likely yield a higher perplexity, say around 10 or more.
x??

---

#### Perplexity as a Model Evaluation Metric
Background context explaining the concept. The text discusses how perplexity is used as an evaluation metric for language models, emphasizing that it reflects the model's uncertainty and predictive accuracy.
:p What does a good value for perplexity depend on?
??x
A good value for perplexity depends on several factors:
- **Data Structure**: More structured data tends to have lower expected perplexity. For example, HTML code is more predictable than everyday text due to its structured nature.
- **Vocabulary Size**: Larger vocabularies result in higher perplexity because there are more possible tokens that the model must predict accurately. Therefore, a children's book with a smaller vocabulary would likely have a lower perplexity compared to War and Peace.
- **Context Length**: Longer context lengths reduce uncertainty, leading to lower perplexity. Modern models can condition their predictions on up to 10,000 previous tokens or more.

In general:
- Perplexity values as low as 3 or even lower are not uncommon for well-performing language models.
x??

---

#### Perplexity as a Model Proxy

Perplexity is used to evaluate the capability of language models, serving as an indirect measure for performance on downstream tasks. Generally, lower perplexity indicates better model performance.

:p What does perplexity indicate about a model's capabilities?
??x
Lower perplexity suggests that a model has better predictive power and is more capable of understanding and generating coherent text. This metric reflects how well the model can predict the next token in a sequence.
x??

---

#### Comparison of Perplexity Across Model Sizes

The OpenAI report shows that larger GPT-2 models consistently give lower perplexity on various datasets, indicating better performance.

:p How does model size affect perplexity according to the OpenAI report?
??x
Larger models tend to have lower perplexity across different datasets, which suggests they are more powerful and capable of better prediction tasks. This trend is observed in Table 3-1 for GPT-2.
x??

---

#### Impact of Post-Training on Perplexity

Post-training techniques like SFT (supervised fine-tuning) and RLHF (reinforcement learning from human feedback) can lead to increased perplexity as models get better at task completion but worse at predicting next tokens.

:p How does post-training affect a model's perplexity?
??x
Post-training can increase a model's perplexity because it improves the model’s performance on specific tasks by making predictions less general. This means that while the model performs well in certain tasks, its ability to predict next tokens may decline.
x??

---

#### Quantization and Perplexity

Quantization, which reduces numerical precision and memory footprint, can also impact a model's perplexity unpredictably.

:p How does quantization affect a model’s perplexity?
??x
Quantization can alter a model's perplexity in unexpected ways. By reducing the numerical precision, it affects how the model processes information, potentially impacting its predictive power.
x??

---

#### Perplexity for Detecting Data Contamination

Perplexity can be used to detect whether a text was part of a model’s training data by comparing the perplexity on known benchmark texts.

:p How can perplexity help in detecting data contamination?
??x
If a model's perplexity is low on a benchmark dataset, it likely means that this dataset was included in the model’s training. This can reduce trust in the model's performance on these benchmarks.
x??

---

#### Deduplication of Training Data Using Perplexity

Perplexity can be used for deduplication: adding new data to the training set only if its perplexity is high, indicating it contains unique information.

:p How does perplexity aid in deduplication?
??x
Adding new data to a training set should be done only if the new data's perplexity is high. This indicates that the new data brings novel content not seen during initial training.
x??

---

#### Perplexity for Detecting Unpredictable Texts

Perplexity is highest for unpredictable texts, like those expressing unusual ideas or gibberish.

:p How does perplexity behave with unpredictable texts?
??x
For unpredictable texts—such as "my dog teaches quantum physics in his free time" or random gibberish—the model's perplexity will be high because it struggles to predict such texts.
x??

---

#### Overall Performance Proxy

Perplexity is a useful proxy for understanding the underlying language model’s performance, which reflects its capabilities on downstream tasks.

:p What role does perplexity play in evaluating models?
??x
Perplexity helps us understand how well a language model can generalize and predict text, serving as a proxy for its overall performance on both training and unseen data.
x??

---

#### Perplexity Calculation for Language Models
Background context: Perplexity is a measure used to evaluate how well a language model predicts a given sequence of tokens. It provides insight into the complexity of generating a particular piece of text. The lower the perplexity, the better the model's performance.
Relevant formula:
$$

P(x_1,x_2,...,x_n) = \left( \prod_{i=1}^{n} \frac{1}{P(x_i|x_1,...,x_{i-1})} \right)^{\frac{1}{n}}$$where $ P(xi|xi−1)$is the probability that the model assigns to the token $ xi$ given the previous tokens.

:p How do you calculate the perplexity of a sequence using a language model?
??x
To calculate the perplexity, you need to compute the product of the reciprocal probabilities of each token in the sequence conditioned on all preceding tokens. This is then raised to the power of $\frac{1}{n}$, where $ n$ is the length of the sequence.
```java
public class PerplexityCalculator {
    public double calculatePerplexity(List<String> tokens, LanguageModel model) {
        double product = 1;
        for (int i = 1; i < tokens.size(); i++) {
            String prevToken = " ".equals(tokens.get(i - 1)) ? "" : tokens.get(i - 1);
            String token = tokens.get(i);
            double prob = model.predictProbability(prevToken, token); // Assume this method exists
            product *= (1 / prob);
        }
        return Math.pow(product, 1.0 / tokens.size());
    }
}
```
x??

---

#### Exact Evaluation vs Subjective Evaluation
Background context: When evaluating models' performance, it is crucial to understand the difference between exact and subjective evaluation methods. Exact evaluations provide clear and unambiguous results, while subjective evaluations depend on human judgment.
Relevant formula: None

:p What are the differences between exact evaluation and subjective evaluation?
??x
Exact evaluation produces judgments without ambiguity, like a correct or incorrect answer in multiple-choice questions. In contrast, subjective evaluation depends on human judgment, such as grading an essay, where different graders might give varying scores even for the same work.
x??

---

#### Functional Correctness Evaluation
Background context: Evaluating systems based on whether they perform intended functionality is known as functional correctness evaluation. This method checks if a model generates output that meets specific criteria, like creating a website or making reservations.
Relevant formula: None

:p How does functional correctness evaluation work?
??x
Functional correctness evaluates systems by checking if the generated outputs meet specified requirements. For example, when generating code for a function `gcd(num1, num2)`, the test involves running the generated code in a Python interpreter to ensure it produces correct results.
x??

---

#### Example of Functional Correctness in Code Generation
Background context: An example of functional correctness evaluation is evaluating a model's ability to generate valid and accurate code. This can be done by checking if the output functions correctly when executed.

:p What does functional correctness mean for code generation tasks?
??x
Functional correctness in code generation means verifying that the generated code not only compiles but also performs its intended function accurately. For instance, a model generating a `gcd` function should produce valid Python code that correctly computes the greatest common divisor.
```java
public class FunctionCorrectnessChecker {
    public boolean checkFunctionCorrectness(String generatedCode) {
        // Use a Java compiler to compile and execute the generated code
        String result = runCompiler(generatedCode); 
        if (result.contains("Exception")) return false;
        int num1 = 24, num2 = 36; // Example input values
        String output = runFunction(num1, num2, generatedCode);
        return Integer.parseInt(output) == gcd(num1, num2); // Verify against known correct result
    }
    
    private String runCompiler(String code) {
        // Pseudo-code for running a compiler to check if the code is valid
    }
    
    private String runFunction(int num1, int num2, String generatedCode) {
        // Pseudo-code for executing the function within the generated code and capturing output
    }

    public static int gcd(int a, int b) {
        return (b == 0) ? a : gcd(b, a % b);
    }
}
```
x??

#### Functional Correctness Evaluation in Code Verification

Background context explaining the concept. Include any relevant formulas or data here.

Functional correctness evaluation is a standard practice in software engineering to ensure that the submitted solutions meet the expected outcomes for given scenarios. This method is widely used by coding platforms like LeetCode and HackerRank, as well as AI benchmarks such as HumanEval and MBPP.

Functional correctness is often validated using unit tests where code runs through different scenarios and checks if it generates the expected outputs. For example, a function to compute the greatest common divisor (GCD) of two numbers should return 5 for the input pair (15, 20).

:p How does functional correctness evaluation work in software engineering?
??x
Functional correctness evaluation involves running code through various test cases to ensure it produces the expected output. For example, a GCD function should correctly return 5 when given inputs 15 and 20.
x??

---

#### Example Test Cases for Function Validation

Background context explaining the concept. Include any relevant formulas or data here.

Test cases are crucial in validating code functionality. They define specific scenarios that the code must handle correctly, along with the expected outcomes. For instance, a function to check if numbers in a list are close to each other within a threshold can be tested using various inputs and thresholds.

:p What is an example of test cases for validating a function's correctness?
??x
Test cases for validating a function include specific scenarios like:
- `has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3)` should return `True`.
- `has_close_elements([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05)` should return `False`.

These test cases help ensure the function works as expected across different scenarios.
x??

---

#### Pass@k Score in Model Evaluation

Background context explaining the concept. Include any relevant formulas or data here.

In model evaluation, especially for AI-generated code, a pass@k score is used to measure how well a model solves problems. The score indicates the fraction of problems that are solved by at least one of k generated code samples out of all problems.

:p How does the pass@k score work in evaluating models?
??x
The pass@k score measures the effectiveness of a model by considering whether any of its k generated code samples pass all test cases for each problem. For example, if a model generates 3 code samples and solves 5 out of 10 problems, the pass@3 score is 50%.

```python
def calculate_pass_at_k(solved_problems: int, total_problems: int, k_samples: int) -> float:
    return solved_problems / (total_problems * k_samples)
```
x??

---

#### Importance of Multiple Code Samples

Background context explaining the concept. Include any relevant formulas or data here.

Increasing the number of code samples generated by a model increases its chances of solving problems correctly, thereby improving the pass@k score. This is because each additional sample provides more opportunities for the model to find a correct solution.

:p Why does increasing the number of code samples improve the pass@k score?
??x
Increasing the number of code samples improves the pass@k score because it gives the model multiple chances to generate a correct solution. With more samples, the probability that at least one sample passes all test cases for each problem increases, leading to better overall performance.

For example, if k = 3 and out of these three samples, two solve different problems, then the pass@3 score is higher than if only one sample solved one problem.
x??

---

#### Chess Game Evaluation Example

Background context explaining the concept. Include any relevant formulas or data here.

Evaluating complex tasks like playing chess can be challenging since end-game outcomes are easier to assess than individual moves. This example illustrates how evaluating a part of a solution might be harder than evaluating the final outcome.

:p How does evaluating individual moves in a chess game compare to evaluating the overall game?
??x
Evaluating individual moves in a chess game is more complex and less straightforward compared to assessing the end-game outcomes (win/lose/draw). The former requires understanding the strategic context, potential counter-moves, and long-term implications of each move. In contrast, the latter provides a clear binary outcome.

For example, evaluating whether a particular move leads to checkmate or forces a draw is intricate but less ambiguous compared to determining if that single move was optimal in a broader strategic sense.
x??

---

---
#### Game Bots Evaluation
Background context: Evaluating game bots, such as a bot for Tetris, can be done by measuring the score it achieves. This is an example of tasks with measurable objectives that can be evaluated using functional correctness.

:p How would you evaluate a Tetris-playing bot?
??x
You would evaluate a Tetris-playing bot by observing its score. The higher the score, the better the performance of the bot. For instance, a bot that clears more lines and holds up the game for longer periods will have a higher score.
x??

---
#### Reference Data Evaluation
Background context: When tasks cannot be evaluated through functional correctness alone, reference data is used to evaluate AI outputs. This method involves comparing generated responses against known correct (reference) responses.

:p What approach would you use if you need to translate sentences from French to English using an AI model?
??x
You would compare the generated English translation with a set of correct English translations provided as reference data. The quality of the generated output is then judged based on how closely it matches these references.
x??

---
#### Similarity Measurements
Background context: When exact match or reference-free methods are not applicable, similarity measurements can be used to evaluate AI outputs. There are three common methods for comparing texts: exact matching, lexical similarity, and semantic similarity.

:p What is the difference between exact matching and lexical similarity?
??x
Exact matching checks if the generated text exactly matches one of the references. Lexical similarity measures how visually similar or closely related the words in the generated text are to those in the reference.
x??

---
#### Semantic Similarity
Background context: Semantic similarity evaluates how close in meaning two texts are, which is particularly useful when exact wording may vary but the essence should be preserved.

:p How do you measure semantic similarity between two open-ended texts?
??x
Semantic similarity measures how closely the generated text aligns with the intended meaning of the reference text. This can involve using natural language processing techniques like word embeddings or machine learning models to compare the meanings.
x??

---
#### AI Evaluators for Similarity
Background context: As exact evaluation methods rely on human-generated reference data, which can be expensive and time-consuming, AI evaluators are increasingly used to reduce this burden.

:p Why might an AI evaluator be preferred over a human evaluator in similarity measurements?
??x
AI evaluators are preferred because they can process vast amounts of data much faster than humans. They also eliminate the variability that comes from different human judgments and reduce costs associated with manual reviews.
x??

---
#### Hand-Designed Metrics
Background context: Exact matching, lexical similarity, and semantic similarity are examples of hand-designed metrics used in evaluating AI outputs.

:p What is a key difference between exact match and the other two metrics?
??x
Exact match checks for an exact textual match, whereas both lexical similarity and semantic similarity measure how closely related or similar the generated text is to the reference, but without requiring an exact match.
x??

---

#### Similarity Measurements for Various Use Cases
Background context: The provided text explains various applications of similarity measurements, including retrieval and search, ranking, clustering, anomaly detection, and data deduplication. These techniques are essential for processing and understanding complex texts or items.
:p What are some use cases mentioned in the text where similarity measurements can be applied?
??x
The text mentions several use cases:
- Retrieval and search: Finding similar items to a query.
- Ranking: Ranking items based on their similarity to a query.
- Clustering: Grouping items that are similar to each other.
- Anomaly detection: Identifying items that are the least similar to others.
- Data deduplication: Removing duplicate or highly similar items.

x??

---

#### Exact Match for Simple Tasks
Background context: The text discusses exact matching, which works well for tasks expecting short and precise answers. This includes simple math problems, common knowledge questions, and trivia-style queries.
:p What is an example of a task where exact match would be appropriate?
??x
An example of a task suitable for exact match is "What’s 2 + 3?" The reference response should be exactly “5”. Any output that precisely matches this will be considered correct.

x??

---

#### Variations on Exact Match Acceptance
Background context: There are variations to the exact match method, such as accepting any output containing a part of the reference response. This is useful when the full response might not always be generated.
:p How does one variation of exact match accept outputs that contain a part of the reference response?
??x
One variation accepts any output that contains parts of the reference response. For instance, in the question "What’s 2 + 3?", if the reference is “5”, the model can output sentences like “The answer is 5” or “2 + 3 is 5” and still be considered correct.

x??

---

#### Challenges with Exact Match for Complex Tasks
Background context: The exact match method struggles with complex tasks due to multiple possible translations or interpretations of longer texts. This makes creating an exhaustive set of reference responses impossible.
:p Why does the exact match method fail in complex tasks?
??x
The exact match method fails in complex tasks because there can be multiple correct translations or interpretations for a given input, making it impractical to create an exhaustive list of all possible responses.

Example: The French sentence “Comment ça va?” has several English translations like "How are you?", "How is everything?", and "How are you doing?". If the model generates "How is it going?", this response would be marked as incorrect despite containing part of the correct answer, because the complete sentence structure is not matched.

x??

---

#### Lexical Similarity
Background context: Lexical similarity measures how much two texts overlap by breaking them into smaller tokens. It can be computed by counting overlapping words or tokens.
:p How does lexical similarity measure the similarity between two texts?
??x
Lexical similarity measures the overlap between two texts by breaking them into tokens (words). The similarity score is calculated based on the number of shared tokens. For example, if a reference response contains 5 unique words and a generated response shares 4 out of those 5 words, it would have an 80% lexical similarity.

Example: Consider:
- Reference Response: "My cats scare the mice"
- Generated Response A: "My cats eat the mice" (4 shared tokens)
- Generated Response B: "Cats and mice fight all the time" (3 shared tokens)

Response A has a higher lexical similarity score because it shares more words with the reference response.

x??

---

#### Semantic Similarity
Background context: Lexical similarity focuses on token overlap, while semantic similarity considers meaning beyond just word choice. It’s particularly useful for complex tasks where exact phrasing might vary.
:p How does semantic similarity differ from lexical similarity?
??x
Semantic similarity differs from lexical similarity by focusing not only on the shared tokens but also on the overall meaning of the text. While lexical similarity measures how many words overlap, semantic similarity evaluates if the meanings align.

Example: Using the same texts:
- Reference Response: "My cats scare the mice"
- Generated Response A: "My cats eat the mice" (Lexical: 80%)
- Generated Response B: "Cats and mice fight all the time" (Lexical: 60%)

Semantically, both responses might be considered more similar because they convey a similar meaning about interaction between cats and mice.

x??

---

#### Edit Distance and Fuzzy Matching
Background context: Lexical similarity can be measured using approximate string matching, known as fuzzy matching. This method calculates the minimum number of single-character edits required to transform one string into another. The three basic operations are deletion, insertion, and substitution.

:p What is edit distance in the context of lexical similarity?
??x
Edit distance measures the minimum number of single-character edits (deletions, insertions, or substitutions) required to change one word or phrase into another.
x??

---

#### Transposition as an Edit Operation
Background context: Some fuzzy matchers consider transpositions, which swap two adjacent letters, as a separate edit operation. However, others treat it as two combined operations—first an insertion and then a deletion.

:p How do some fuzzy matchers handle the transposition of two characters?
??x
Some fuzzy matchers treat a transposition (swapping two adjacent letters) as one edit operation. Others break it down into two separate operations: first, an insertion, followed by a deletion.
x??

---

#### Token Processing for Fuzzy Matching
Background context: In some cases, you might need to process tokens depending on how "cats" and "cat" are treated or whether "will not" and "won't" should be considered as two separate words. This can affect the edit distance calculation.

:p How does token processing impact fuzzy matching?
??x
Token processing impacts fuzzy matching by determining if certain forms of words, like contractions ("won't"), are considered a single token or split into multiple tokens. This can influence how edits are counted.
x??

---

#### N-Gram Similarity Measure
Background context: An alternative to edit distance is n-gram similarity, which measures the overlap of sequences of tokens (n-grams) between two texts. A bigram consists of two consecutive tokens.

:p What is an n-gram and how does it differ from edit distance?
??x
An n-gram is a contiguous sequence of n items from a given sample of text or speech. In contrast to edit distance, which focuses on single character operations, n-grams consider sequences of multiple tokens for similarity measurement.
x??

---

#### Common Metrics for Lexical Similarity
Background context: Various metrics like BLEU, ROUGE, METEOR++, TER, and CIDEr are used to measure lexical similarity. These metrics differ in how they calculate the overlap of tokens or n-grams.

:p What are some common metrics used to measure lexical similarity?
??x
Common metrics for measuring lexical similarity include BLEU, ROUGE, METEOR++, TER, and CIDEr. Each metric calculates the overlap differently.
x??

---

#### Challenges with Reference-Based Metrics
Background context: While reference-based metrics like BLEU, ROUGE, etc., are useful, they face challenges such as requiring a comprehensive set of reference responses and potential biases from low-quality references.

:p What are some drawbacks of using reference-based metrics for measuring lexical similarity?
??x
Drawbacks include the need to curate a comprehensive set of high-quality reference responses. A good response might receive a low score if no similar responses exist in the reference data. Additionally, low-quality or incorrect references can lead to misleading scores.
x??

---

#### Example Benchmark and Evaluation
Background context: Some benchmarks like WMT, COCO Captions, and GEMv2 use these metrics for evaluation. However, issues with reference quality and other factors make them less effective in some scenarios.

:p Which benchmark examples use reference-based metrics?
??x
Examples of benchmarks that use reference-based metrics include WMT (for machine translation), COCO Captions, and GEMv2.
x??

---

#### Limitations of Lexical Similarity Metrics
Background context: Higher lexical similarity scores do not always correlate with better quality responses. For instance, in code generation tasks like HumanEval, optimizing for BLEU scores might not align with the goal of generating functionally correct solutions.

:p Why might optimizing for lexical similarity metrics like BLEU not be effective?
??x
Optimizing for lexical similarity metrics such as BLEU doesn't necessarily correlate with functional correctness. In tasks like code generation, incorrect and correct solutions can have similar BLEU scores, indicating that these metrics don't ensure the quality of generated content.
x??

---

#### Semantic Similarity vs Lexical Similarity

Background context explaining the concept. Include any relevant formulas or data here.

Lexical similarity measures whether two texts look similar, not whether they have the same meaning. Consider the two sentences “What’s up?” and “How are you?” Lexically, they are different—there’s little overlapping in the words and letters they use. However, semantically, they are close. Conversely, similar-looking texts can mean very different things. “Let’s eat, grandma” and “Let’s eat grandma” mean two completely different things.

:p What is semantic similarity?
??x
Semantic similarity aims to compute the similarity in semantics.
x??

---

#### Semantic Similarity and Embeddings

Background context explaining the concept. Include any relevant formulas or data here.

For computing semantic similarity, texts are first transformed into numerical representations called embeddings. For example, the sentence “the cat sits on a mat” might be represented using an embedding that looks like this: [0.11, 0.02, 0.54]. Semantic similarity is therefore also called embedding similarity.

:p What is an embedding?
??x
An embedding is a numerical representation of text.
x??

---

#### Cosine Similarity Calculation

Background context explaining the concept. Include any relevant formulas or data here.

The similarity between two embeddings can be computed using metrics such as cosine similarity. Two embeddings that are exactly the same have a similarity score of 1. Two opposite embeddings have a similarity score of –1. Mathematically, let A be an embedding of the generated response, and B be an embedding of a reference response. The cosine similarity between A and B is computed as $\frac{A·B}{||A|| ||B||}$, with:
- $A·B$ being the dot product of A and B
- $||A||$ being the Euclidean norm (also known as L2 norm) of A.

:p How do you calculate cosine similarity between two embeddings?
??x
The cosine similarity is calculated by taking the dot product of the two vectors and dividing it by the product of their Euclidean norms. Here's an example in pseudocode:

```pseudocode
function cosineSimilarity(A, B):
    dotProduct = 0.0
    normA = 0.0
    normB = 0.0

    for i from 0 to length(A)-1:
        dotProduct += A[i] * B[i]
        normA += A[i]^2
        normB += B[i]^2

    normA = sqrt(normA)
    normB = sqrt(normB)

    return dotProduct / (normA * normB)
```

x??

---

#### Semantic Textual Similarity Metrics

Background context explaining the concept. Include any relevant formulas or data here.

Semantic textual similarity is computed using embeddings, and metrics such as BERTScore (embeddings are generated by BERT) and MoverScore (embeddings are generated by a mixture of algorithms). While semantic similarity can be considered subjective due to different embedding algorithms producing different results, the score between two given embeddings is computed exactly.

:p What are some metrics used for semantic textual similarity?
??x
Some metrics used for semantic textual similarity include BERTScore and MoverScore.
x??

---

#### Importance of Embeddings

Background context explaining the concept. Include any relevant formulas or data here.

The quality of the underlying embedding algorithm significantly impacts the reliability of semantic similarity. Two texts with the same meaning can still have a low semantic similarity score if their embeddings are poor. Additionally, running an embedding algorithm might require substantial compute and time.

:p Why is the quality of the embedding important?
??x
The quality of the embedding is crucial because it directly affects the accuracy of the semantic similarity score. Poor embeddings may lead to incorrect or misleading similarity scores even when the texts have the same meaning.
x??

---

