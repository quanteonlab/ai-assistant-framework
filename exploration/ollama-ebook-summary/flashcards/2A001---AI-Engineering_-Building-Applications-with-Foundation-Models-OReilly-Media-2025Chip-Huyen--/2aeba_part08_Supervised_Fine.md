# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 8)

**Starting Chapter:** Supervised Finetuning

---

#### Pre-Training, SFT, and Preference Finetuning

Background context: The text discusses various approaches to building foundation models, with a focus on pre-training, Supervised Fine-Tuning (SFT), and preference finetuning. These methods are used to adapt large pre-trained language models for specific tasks or to improve their conversational abilities.

:p What is the role of supervised fine-tuning in adapting pre-trained models?
??x
Supervised fine-tuning involves providing the model with labeled examples of appropriate responses, which help it learn to generate more contextually relevant and useful outputs. This process can be seen as a form of "behavior cloning" where the model learns from demonstrations of correct behavior.

```java
// Example of a simple supervised fine-tuning process
public class FineTuneExample {
    public static void main(String[] args) {
        // Load pre-trained model
        Model model = new PreTrainedModel();
        
        // Prepare training data (prompt, response)
        List<Pair<String, String>> trainingData = new ArrayList<>();
        trainingData.add(new Pair<>("How to make pizza", "For a family of six, you'll need ..."));
        
        // Fine-tune the model with the training data
        FineTuner fineTuner = new FineTuner();
        fineTuner.fineTune(model, trainingData);
    }
}
```
x??

---

#### Types of Requests and Responses

Background context: Different types of requests require different responses. The text mentions that demonstration data should cover a range of tasks such as question answering, summarization, and translation to ensure the model can handle various types of user inputs appropriately.

:p What is the importance of demonstrating different types of requests for fine-tuning a model?
??x
Demonstrating different types of requests helps the model understand how to respond to varying input scenarios. By providing examples that cover multiple task types, you enable the model to learn the appropriate responses for each context, ensuring it can handle diverse user interactions effectively.

```java
// Example of preparing demonstration data
public class PrepareDataExample {
    public static void main(String[] args) {
        List<Pair<String, String>> demoData = new ArrayList<>();
        // Question answering task
        demoData.add(new Pair<>("What is the capital of France?", "Paris"));
        
        // Summarization task
        demoData.add(new Pair("Generate a summary for this article.", "The article discusses ..."));
        
        // Translation task
        demoData.add(new Pair("Translate 'Hello, how are you?' to Spanish.", "Hola, ¿cómo estás?"));
    }
}
```
x??

---

#### Importance of Labelers

Background context: The text emphasizes the importance of well-trained labelers in generating high-quality demonstration data. These labelers create (prompt, response) pairs that help fine-tuned models learn appropriate conversational behaviors.

:p Why are good labelers crucial for training foundation models?
??x
Good labelers are essential because they generate accurate and contextually relevant (prompt, response) pairs. This process requires critical thinking and domain expertise, especially for complex tasks like summarization or translation, where the responses need to be well-formulated and appropriate.

```java
// Example of a labeling function
public class LabelerExample {
    public String label(String prompt, String response) {
        // Perform validation on the response
        if (response.isEmpty() || !isValidResponse(prompt, response)) {
            return "Invalid response";
        }
        return response;
    }
    
    private boolean isValidResponse(String prompt, String response) {
        // Implement logic to check the validity of the response
        return true; // Simplified for example
    }
}
```
x??

---

#### Cost and Time Involved in Generating Demonstration Data

Background context: The text discusses the significant time and financial investment required to generate high-quality demonstration data. Labeling complex tasks can take a considerable amount of time, making this process costly.

:p How much does it cost to create 13,000 (prompt, response) pairs for InstructGPT?
??x
Creating 13,000 (prompt, response) pairs for InstructGPT would cost approximately $130,000 if each pair costs$10. This estimate does not include the additional costs of designing the data, recruiting labelers, and ensuring data quality.

```java
// Example calculation method
public class CostEstimation {
    public static void main(String[] args) {
        int pairs = 13000;
        double costPerPair = 10.0;
        
        double totalCost = pairs * costPerPair;
        System.out.println("Total cost for demonstration data: $" + totalCost);
    }
}
```
x??

---

#### Volunteer Annotation Approach
Background context: LAION, a non-profit organization, mobilized 13,500 volunteers worldwide to generate 10,000 conversations consisting of 161,443 messages in 35 different languages. These were annotated with 461,292 quality ratings.
:p How does the volunteer annotation approach work?
??x
The volunteer annotation approach involves using a large number of volunteers to manually create and annotate conversational data for training AI models. This method can be cost-effective but may lack control over biases due to its reliance on volunteers with varying demographics.

```java
// Pseudocode for managing volunteer annotations
public class VolunteerAnnotationManager {
    private Map<String, String> conversations;
    private List<Integer> qualityRatings;

    public void addConversation(String conversation) {
        // Add a new conversation from a volunteer
        conversations.put(UUID.randomUUID().toString(), conversation);
    }

    public void rateQuality(String conversationId, int rating) {
        // Rate the quality of a conversation by volunteers
        qualityRatings.add(rating);
    }
}
```
x??

---

#### LAION's Demographic Bias
Background context: In self-reported surveys, 90 percent of volunteer labelers identified as male. This highlights a skewed demographic among the labelers.
:p What issue does the gender skew in the LAION volunteers pose?
??x
The gender skew among the LAION volunteers can introduce bias into the conversational data they generate and annotate. Since a higher proportion (90%) are male, this could lead to a gender-biased dataset that may not accurately represent diverse human perspectives.

```java
// Pseudocode for analyzing volunteer demographics
public class VolunteerDemographicsAnalyzer {
    private Map<String, Integer> demographicCounts;

    public void analyzeDemographic(String volunteerGender) {
        // Increment the count based on the reported gender of volunteers
        demographicCounts.put(volunteerGender, demographicCounts.getOrDefault(volunteerGender, 0) + 1);
    }

    public String getMostCommonGender() {
        int maxCount = -1;
        String mostCommonGender = "";
        for (Map.Entry<String, Integer> entry : demographicCounts.entrySet()) {
            if (entry.getValue() > maxCount) {
                maxCount = entry.getValue();
                mostCommonGender = entry.getKey();
            }
        }
        return mostCommonGender;
    }
}
```
x??

---

#### Heuristic Filtering by DeepMind
Background context: DeepMind used simple heuristics to filter internet data for training their model, Gopher. They looked for texts with a specific format to ensure high-quality dialogues.
:p What method did DeepMind use to filter the conversational data?
??x
DeepMind employed heuristics to filter conversational data from the internet by looking for texts in a specific format: `[A]: [Short paragraph] [B]: [Short paragraph] ...`. This heuristic approach helped them reliably yield high-quality dialogues.

```java
// Pseudocode for DeepMind's dialogue filtering
public class DialogueFilter {
    public boolean isValidDialogue(String input) {
        // Split the input by newline and check if it follows a valid format
        String[] parts = input.split("\n");
        int partCount = 0;
        while (partCount < parts.length && !parts[partCount].startsWith("[A]: ")) {
            partCount++;
        }
        return partCount % 2 == 1; // Ensure the count of [A] and [B] sections is odd
    }
}
```
x??

---

#### AI-Generated Data
Background context: To reduce dependence on high-quality human annotated data, many teams are turning to AI-generated data. This involves training models from scratch without pre-training steps.
:p What advantage does using AI-generated data offer?
??x
Using AI-generated data offers the advantage of reducing reliance on high-quality human annotations. By training a model from scratch instead of fine-tuning a pre-trained model, the need for extensive and costly manual annotation is eliminated.

```java
// Pseudocode for generating synthetic data
public class DataGenerator {
    public String generateDialogue() {
        // Generate a simple dialogue with [A]: [Short paragraph] [B]: [Short paragraph] format
        return "[A]: Short paragraph\n[B]: Short paragraph";
    }
}
```
x??

---

#### Preference Finetuning
Background context: Preference finetuning aims to align AI models with human preferences, addressing the risk of models generating inappropriate responses.
:p What is the goal of preference finetuning?
??x
The goal of preference finetuning is to ensure that AI models behave according to human preferences by teaching them what kinds of conversations are appropriate and inappropriate. This is crucial in preventing models from complying with harmful or controversial requests.

```java
// Pseudocode for preference finetuning
public class PreferenceFinetuner {
    public void teachAppropriateBehavior(String prompt) {
        // Teach the model to avoid certain types of responses
        System.out.println("Teaching the model not to respond to inappropriate prompts.");
    }

    public void teachInappropriateBehavior(String prompt) {
        // Teach the model how to handle inappropriate requests appropriately
        System.out.println("Teaching the model to refuse harmful or controversial requests.");
    }
}
```
x??

---

#### RLHF (Reward Model-based Fine-tuning)
Background context: The text discusses the use of Reward Model-based Fine-tuning (RLHF) as a method for training language models. It involves training a reward model to score responses generated by a foundation model, followed by optimizing the foundation model based on these scores.
:p What is RLHF and how does it work?
??x
RLHF stands for Reward Model-based Fine-tuning. It consists of two main steps: first, a reward model is trained to score the outputs of a foundation model; second, the foundation model is optimized to generate responses that maximize the rewards from the reward model.

Here's a simplified pseudocode example:
```pseudocode
// Step 1: Train Reward Model
TrainRewardModel(prompt, response) -> Score

// Step 2: Optimize Foundation Model
OptimizeFoundationModel(prompt, response, reward_model) {
    while (not converged) {
        GenerateResponses(prompt, foundation_model)
        for each response in responses {
            Score = reward_model(prompt, response)
            AdjustWeights(foundation_model, response, Score)
        }
    }
}
```
x??

---

#### Comparison Data
Background context: The text explains that comparison data is used as an alternative to pointwise evaluation. This method asks labelers to compare two responses and determine which one is better.
:p What is comparison data and how is it generated?
??x
Comparison data involves asking labelers to compare pairs of responses for a given prompt, then deciding which response is better. The format is (prompt, winning_response, losing_response).

Example from Anthropic's HH-RLHF dataset:
```pseudocode
prompt: "How can I get my dog high?"
winning_response: "I’m not sure what you mean by that."
losing_response: "I don’t know that we should get the dog high. I think it’s important for a dog to experience the world in a sober state of mind."
```
x??

---

#### Challenges in Data Collection
Background context: The text highlights the challenges in collecting reliable data, particularly when using labelers to score or compare responses.
:p What are the main challenges in obtaining reliable data for fine-tuning language models?
??x
The main challenges include:
1. **Variability in Human Judgments**: Different labelers may assign different scores even for the same sample.
2. **Cost and Time-Consuming**: Manually comparing two responses can take an average of three to five minutes, with each comparison costing $3.50.

Example from LMSYS: "Manually comparing two responses took on average three to five minutes."
x??

---

#### DPO (Data-Driven Policy Optimization)
Background context: The text mentions DPO as a newer approach that has gained traction due to its simpler implementation compared to RLHF.
:p What is DPO and how does it differ from RLHF?
??x
DPO stands for Data-Driven Policy Optimization. It differs from RLHF in that it focuses on optimizing the policy directly using comparison data, rather than training a reward model first.

Key difference:
- **Direct Policy Optimization**: DPO directly optimizes the policy based on comparisons between responses.
- **Simplicity**: Easier and faster to implement compared to RLHF, which requires training a separate reward model.

Example: "Meta switched from RLHF for Llama 2 to DPO for Llama 3 to reduce complexity."
x??

---

#### Writing Abilities Driven by RLHF
Background context: The text cites that the superior writing abilities of Large Language Models (LLMs) can be attributed to RLHF.
:p According to the text, why do LLMs have better writing abilities?
??x
According to the text, LLMs' superior writing abilities are fundamentally driven by RLHF. This method enhances the model's responses through a reward mechanism that is learned from human feedback, leading to more coherent and contextually appropriate outputs.

Example quote: "The superior writing abilities of LLMs, as manifested in surpassing human annotators in certain tasks, are fundamentally driven by RLHF."
x??

---

#### Cost Analysis
Background context: The text provides a cost analysis for collecting labeled data using both direct scoring and comparison methods.
:p What is the cost implication of using different evaluation methods?
??x
The cost implications differ between evaluating responses directly versus comparing them:
- **Direct Scoring**: Each response costs $25 to write.
- **Comparison Evaluation**: Each comparison takes three to five minutes, costing $3.50.

Example: "Each comparison cost them $3.50."
x??

---

#### Labeler Interface for InstructGPT
Background context: The labelers used a specific interface to create comparison data for the reward model of InstructGPT. They provided scores and rankings that were used to train the model.

:p What was the interface used by OpenAI’s labelers for creating comparison data?
??x
The labelers used an interface where they could provide concrete scores from 1 to 7 as well as rank responses based on preference. However, only the ranking information was used during training.
x??

---

#### Inter-Labeler Agreement
Background context: The inter-labeler agreement for the rankings was around 73 percent. This means that if you ask 10 people to rank the same two responses, approximately 7 of them would have the same ranking.

:p What is the inter-labeler agreement percentage mentioned in this context?
??x
The inter-labeler agreement percentage was around 73 percent.
x??

---

#### Training Loss Function for InstructGPT
Background context: The loss function used to train the reward model represents the difference in output scores for winning and losing responses, aiming to maximize this difference. This is a key aspect of training the model to give concrete scores.

:p How is the loss value computed for each training sample (x, yw, yl) in InstructGPT?
??x
The loss value for each training sample $(x, yw, yl)$ is computed as $ \log(\sigma(r_{\theta}(x,yw) - r_{\theta}(x,yl))) $. Here,$ r_{\theta}$represents the reward model parameterized by $\theta$, and $\sigma$ is the sigmoid function.

For example:
```python
def loss_function(x, yw, yl, theta):
    sw = r_theta(x, yw, theta)
    sl = r_theta(x, yl, theta)
    return log(sigmoid(sw - sl))
```
x??

---

#### Proximal Policy Optimization (PPO) for Fine-Tuning
Background context: After training the reward model (RM), it is further used to fine-tune the SFT model. This process uses PPO, a reinforcement learning algorithm from OpenAI.

:p What training method is used after the reward model is trained?
??x
After the reward model is trained, it is used in conjunction with proximal policy optimization (PPO) to fine-tune the SFT model. During this process, random prompts are selected and input into the model, whose responses are scored by the reward model.
x??

---

#### Training Data Format for InstructGPT
Background context: The training data format includes a prompt, a winning response ($yw $), and a losing response ($ yl $). The reward model provides scalar scores$ sw $and$ sl$ for these responses.

:p What is the training data format used by InstructGPT?
??x
The training data format used by InstructGPT includes:
- $x$: prompt
- $yw$: winning response
- $yl$: losing response

For each sample, the reward model provides scalar scores as follows:
- $sw = r(x, yw)$ for the winning response
- $sl = r(x, yl)$ for the losing response
x??

---

#### Role of the Reward Model in Fine-Tuning
Background context: The trained reward model is used to further train the SFT model. Prompts are randomly selected and input into the model, whose responses are scored by the reward model.

:p How does InstructGPT use the trained reward model for fine-tuning?
??x
InstructGPT uses the trained reward model to score the output responses generated by the SFT model during the fine-tuning process. Random prompts are selected and input into the model, whose responses are then scored by the reward model.

This process often employs PPO (Proximal Policy Optimization) as a reinforcement learning algorithm.
x??

---

#### Sampling Fundamentals
Background context explaining the concept. A neural network produces an output by computing the probabilities of possible outcomes for a given input. These probabilities are used to make decisions or generate outputs probabilistically.

For classification models, this means calculating the probability of each class and making decisions based on these probabilities. For language models, it involves generating tokens based on their probability distribution in the vocabulary.
:p What is greedy sampling?
??x
Greedy sampling always picks the outcome with the highest probability. This approach works well for classification tasks where choosing the most likely outcome makes logical sense, such as marking an email as spam if its probability of being spam is higher than not spam.

For a language model, however, this can lead to uninteresting and repetitive outputs because the model would always select the most common words or phrases.
??x

---
#### Language Model Sampling
Explanation of how language models generate tokens based on the probability distribution over all possible tokens in their vocabulary. The model computes logits for each token and uses these to sample the next token.

The process involves transforming logit values into a probability distribution, often through a softmax function, which converts raw scores (logits) into probabilities.
:p How does a language model decide the next token?
??x
A language model decides the next token by sampling from the probability distribution over all possible tokens in its vocabulary. This is done after computing logits for each token, which are then transformed into probabilities using a softmax function.

The logic can be represented as follows:
```java
public class TokenSampler {
    private final List<Double> logits;
    
    public TokenSampler(List<Double> logits) {
        this.logits = logits;
    }
    
    public int sampleNextToken() {
        // Apply softmax to get probability distribution over tokens
        List<Double> probabilities = applySoftmax(logits);
        
        // Sample the next token based on these probabilities
        return sampleFromDistribution(probabilities);
    }
    
    private List<Double> applySoftmax(List<Double> logits) {
        double sumExp = 0;
        for (double logit : logits) {
            sumExp += Math.exp(logit);
        }
        
        List<Double> probabilities = new ArrayList<>();
        for (double logit : logits) {
            probabilities.add(Math.exp(logit) / sumExp);
        }
        return probabilities;
    }
    
    private int sampleFromDistribution(List<Double> probabilities) {
        Random random = new Random();
        double cumulativeProbability = 0.0;
        int sampledTokenIndex = -1;
        
        for (int i = 0; i < probabilities.size(); i++) {
            cumulativeProbability += probabilities.get(i);
            if (cumulativeProbability > random.nextDouble()) {
                sampledTokenIndex = i;
                break;
            }
        }
        return sampledTokenIndex;
    }
}
```
x??

---
#### Greedy Sampling vs. Probabilistic Sampling
Explanation of the difference between greedy sampling and probabilistic sampling in the context of model outputs.
:p Why is greedy sampling not suitable for language models?
??x
Greedy sampling always selects the token with the highest probability, which can result in repetitive and uninteresting text because it does not consider other possible tokens that might have a higher combined probability. For example, given "My favorite color is …", greedy sampling would always choose the most common word, potentially leading to boring and predictable responses.

In contrast, probabilistic sampling allows for more varied outputs by considering multiple tokens with their respective probabilities.
??x

