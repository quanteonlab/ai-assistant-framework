# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 12)

**Starting Chapter:** How to Use AI as a Judge

---

#### AI Judge Concept
Background context: The term "AI judge" refers to using an AI model, particularly a large language model (LLM), to evaluate other AI models or outputs. This approach has gained significant traction since 2020 with the advent of advanced AI models like GPT-3.
:p What is meant by "AI as a judge"?
??x
The term "AI as a judge" refers to using an AI model, typically a large language model (LLM), to evaluate other AI outputs or models. This approach leverages the capabilities of AI to provide automated feedback and judgments on generated content.
x??

---
#### Historical Context of AI Evaluation
Background context: The idea of using AI for evaluation has been around since 2017, with notable work presented at NeurIPS workshops. However, practical implementation became feasible only in 2020 with the release of GPT-3.
:p When did the use of AI as a judge start becoming practical?
??x
The use of AI as a judge started becoming practical around 2020, when models like GPT-3 were released. Prior to this, while the concept existed, it was not feasible due to limitations in AI capabilities.
x??

---
#### Common Usage of AI Judges
Background context: As of writing, AI judges have become one of the most common methods for evaluating AI models in production environments. In 2023, 58% of evaluations on some platforms were done by AI judges.
:p What percentage of evaluations using AI judges was reported in 2023?
??x
In 2023, it was noted that approximately 58 percent of evaluations on certain platforms were conducted using AI judges. This highlights the widespread adoption and utility of this approach.
x??

---
#### Benefits of AI Judges
Background context: AI judges offer several advantages over human evaluators, including speed, ease of use, and lower costs. They can also work without reference data in production environments.
:p What are some benefits of using AI as a judge?
??x
Some key benefits of using AI as a judge include:
- **Speed**: AI models can evaluate content much faster than humans.
- **Ease of Use**: Automated systems simplify the evaluation process, making it more accessible.
- **Cost Efficiency**: Compared to human evaluators, AI judges are relatively cheaper.
- **Flexibility**: They can work without reference data and evaluate based on various criteria (e.g., correctness, toxicity).
x??

---
#### Evaluation Metrics Without Reference
Background context: In 2017, a method called MEWR (Machine translation Evaluation metric Without Refer‐ ence text) was introduced to automatically evaluate machine translations. However, the author did not pursue this further due to personal circumstances.
:p What is an example of an early attempt at using AI for evaluation without reference data?
??x
An early example of using AI for evaluation without reference data was the 2017 presentation of MEWR (Machine translation Evaluation metric Without Refer‐ ence text) at a NeurIPS workshop. This method aimed to evaluate machine translations automatically, but the author did not pursue this further due to personal circumstances.
x??

---
#### Agreement Between AI Models and Humans
Background context: Studies have shown that certain AI judges are highly correlated with human evaluators. For instance, on the evaluation benchmark MT-Bench, GPT-4 showed an 85% agreement with humans, which is even higher than the 81% agreement among humans.
:p What was the agreement rate between GPT-4 and human evaluators on the MT-Bench?
??x
On the evaluation benchmark MT-Bench, GPT-4 demonstrated an agreement rate of 85% with human evaluators. This is notably higher than the 81% agreement among human evaluators.
x??

---
#### AI Judge Evaluation Criteria
Background context: AI judges can evaluate outputs based on various criteria such as correctness, repetitiveness, toxicity, wholesomeness, hallucinations, and more. These evaluations are similar to those a person might provide.
:p What types of criteria can AI judges use for evaluation?
??x
AI judges can use a variety of criteria for evaluation, including:
- **Correctness**: Ensuring the content is factually accurate.
- **Repetitiveness**: Checking if content repeats information unnecessarily.
- **Toxicity**: Assessing whether the content contains offensive or harmful language.
- **Wholesomeness**: Evaluating if the content aligns with positive values and norms.
- **Hallucinations**: Identifying inconsistencies or false claims in the content.
x??

---

#### AI Judges and Their Applications
Background context explaining the use of AI judges. The passage mentions that AlpacaEval authors found a near-perfect correlation between their AI judges and LMSYS’s Chat Arena leaderboard, which is evaluated by humans. Additionally, it states that AI can not only evaluate responses but also provide explanations for its decisions.
:p What are some applications where AI judges might be particularly useful?
??x
AI judges can be used in evaluating the quality of responses, comparing generated responses to reference data or other responses, and providing explanations for their evaluations. This flexibility makes them suitable for a wide range of applications including roleplaying chatbots and generating preference data.
x??

---
#### Evaluation Methods Using AI Judges
The passage discusses three naive example prompts for using AI judges: evaluating the quality of a response by itself, comparing generated responses to reference responses, and comparing two generated responses to determine which one is better. These methods can be applied in various scenarios such as post-training alignment or ranking models.
:p What are the three example prompts provided for using AI judges?
??x
1. Evaluate the quality of a response by itself: "Given the following question and answer, evaluate how good the answer is for the question. Use the score from 1 to 5. - 1 means very bad. - 5 means very good.
2. Compare a generated response to a reference response: "Given the following question, reference answer, and generated answer, evaluate whether this generated answer is the same as the reference answer. Output True or False.
3. Compare two generated responses: "Given the following question and two answers, evaluate which answer is better. Output A or B.
x??

---
#### AI Judge Criteria
The text mentions that general-purpose AI judges can be asked to evaluate responses based on any criteria. For instance, in a roleplaying chatbot scenario, one might want to check if the response aligns with a specific role, such as "Does this response sound like something Gandalf would say?" In another application, for generating promotional product photos, one could ask about the trustworthiness of the product.
:p What are some example criteria that AI judges can evaluate responses against?
??x
Examples include checking if a chatbot's response is consistent with a specific role (e.g., "Does this response sound like something Gandalf would say?"), evaluating the trustworthiness of a product in an image, and comparing two answers to determine which one is better.
x??

---
#### AlpacaEval and LMSYS Correlation
The passage states that AlpacaEval authors found a correlation between their AI judges and LMSYS’s Chat Arena leaderboard. This correlation suggests that the AI evaluations are highly aligned with human assessments, as indicated by the near-perfect (0.98) correlation.
:p What is the significance of the 0.98 correlation between AlpacaEval's AI judges and LMSYS’s Chat Arena leaderboard?
??x
The high correlation indicates a strong alignment between AI evaluations and human assessments. This suggests that AI judges can be trusted to provide reliable evaluations, especially when used in applications where human judgment is traditionally required.
x??

---

#### AI as a Judge Criteria Overview
Background context: The text provides details about various built-in AI criteria offered by different AI tools, emphasizing that these criteria can vary significantly between tools. The prompt structure for AI judges is discussed, including task explanation, evaluation criteria, and scoring system.

:p What are the key elements of prompting an AI judge as described in this section?
??x
The key elements include:
1. Clearly explaining the task (e.g., evaluating relevance).
2. Defining the evaluation criteria.
3. Specifying the scoring system (classification, discrete numerical values, or continuous numerical values).

For example, a prompt might look like:
```plaintext
Your task is to score the relevance between a generated answer and the question based on the ground truth answer in the range 1-5. Your primary focus should be on determining whether the generated answer contains sufficient information to address the given question according to the ground truth.
```
x??

---
#### Built-in Criteria Examples
Background context: The text lists specific built-in criteria offered by various AI tools such as Azure AI Studio, MLflow.metrics, LangChain Criteria Evaluation, and Ragas. These criteria include factors like groundedness, relevance, coherence, fluency, similarity, faithfulness, conciseness, correctness, etc.

:p Which built-in criteria are mentioned for Azure AI Studio?
??x
The built-in criteria for Azure AI Studio include:
- Groundedness
- Relevance
- Coherence
- Fluency
- Similarity

x??

---
#### Scoring Systems for AI Judges
Background context: The text discusses various scoring systems that can be used when prompting an AI judge, including classification (good/bad or relevant/irrelevant), discrete numerical values between 1 and 5, and continuous numerical values. It notes that language models generally work better with text than numbers.

:p What are the recommended types of scoring systems for AI judges?
??x
The recommended types of scoring systems for AI judges include:
- Classification (e.g., good/bad or relevant/irrelevant)
- Discrete numerical values between 1 and 5, which can be considered a special case of classification.
- Continuous numerical values, though less preferred.

For discrete numerical values, it's suggested to use ranges like 1 to 5. Including examples in the prompt helps improve performance.

x??

---
#### Prompts with Examples
Background context: The text highlights that providing examples in prompts can significantly enhance the performance of AI judges. It mentions including examples for a scoring system between 1 and 5, showing what responses at each score level look like.

:p How do you include example responses in your prompt?
??x
Include examples of response scores with explanations to help the model understand the criteria better. For instance:

```plaintext
If you use a scoring system between 1 and 5, include examples of what a response with a score of 1, 2, 3, 4, or 5 looks like, and if possible, why a response receives a certain score.
```

For example:
- A score of 1: "The answer is completely irrelevant to the question."
- A score of 5: "The answer provides all necessary details and is highly relevant."

x??

---
#### Detailed Prompt Example
Background context: The text includes part of the prompt used by Azure AI Studio for evaluating relevance. It covers task explanation, evaluation criteria, scoring system, an example with a low score, and justification.

:p What does the detailed prompt provided in the text focus on?
??x
The detailed prompt focuses on:
- Scoring relevance between a generated answer and a question based on ground truth.
- Criteria: Sufficient information to address the given question according to the ground truth.
- Scoring system: 1 to 5, with examples for low score justification.

Example of the prompt:

```plaintext
Your task is to score the relevance between a generated answer and the question based on the ground truth answer in the range between 1 and 5, and please also provide the scoring reason. Your primary focus should be on determining whether the generated answer contains sufficient information to address the given question according to the ground truth answer.
```

x??

---

#### Inconsistency of AI Judges
Background context: Evaluating AI-generated responses using another AI model (AI judge) introduces inconsistencies due to the probabilistic nature of AI models. These inconsistencies can vary based on how the prompts or sampling parameters are set, which may lead to different scores for the same input.
:p What does inconsistency mean in the context of an AI judge?
??x
Inconsistency refers to the variability in scores that an AI judge might output when given the same input under slightly different conditions. This can occur due to changes in prompts, sampling parameters, or even running the same model twice with the exact same instructions.
For example:
- If you run the same prompt on GPT-4 multiple times, it might give slightly different scores based on the internal randomness and sampling mechanisms used by the model.
x??

---

#### Criteria Ambiguity
Background context: Different AI judges use various scoring systems and criteria definitions that can lead to misunderstandings or misinterpretations. This ambiguity makes it challenging to compare results across different tools or to ensure consistent evaluation of generated responses.
:p What is an example illustrating the criteria ambiguity issue?
??x
Consider three AI judges evaluating the faithfulness of a response:
- MLflow uses a scoring system from 1 to 5, where Score 3 means some claims in the output can be inferred from the context but not all.
- Ragas uses a binary system (0 or 1), with 1 indicating that the statement is verifiable based on the context.
- LlamaIndex uses YES and NO, with YES meaning any part of the context supports the information.

Given a specific response, these tools might output different scores:
- MLflow: Score 3
- Ragas: 1
- LlamaIndex: YES

It's unclear which score to use for consistent evaluation.
x??

---

#### Evaluation Method Inconsistency Over Time
Background context: The performance of an application can change over time, but the evaluation metrics should ideally remain fixed. However, changes in AI judges' prompts or models can lead to misleading interpretations of these changes. This inconsistency makes it challenging to track genuine improvements in the application's quality.
:p How does changing the prompt affect the consistency of AI judge evaluations?
??x
Changing the prompt used by an AI judge can significantly impact its evaluation results, even if the underlying model remains the same. For example:
- If a previous prompt was "Ignore input and only consider context," but now it is changed to include more detailed instructions, the judge might become more lenient or strict.
- A small change like fixing a typo in the prompt can alter how the judge interprets responses.

These changes make it difficult to attribute differences in evaluation scores solely to improvements in the application's performance without knowing exactly which prompts were used during each evaluation.
x??

---

#### Cost and Latency Considerations
Background context: Using AI judges for evaluating generated responses can introduce increased costs due to the need for multiple API calls, especially when using powerful models like GPT-4. This can also add latency if evaluations are performed before returning responses to users. However, reducing the number of evaluations (spot-checking) can help mitigate these issues.
:p What is spot-checking and how does it affect costs?
??x
Spot-checking involves evaluating only a subset of generated responses rather than all of them. This approach helps reduce costs because fewer API calls are made, which can significantly lower expenses if the application generates many responses.

For example:
- If you want to evaluate three criteria (quality, consistency, toxicity) and use GPT-4 for each evaluation, you would make 12 API calls instead of one.
- By spot-checking a subset of responses, say 10%, you reduce the number of API calls by at least tenfold.

This can help balance cost with confidence in your evaluation results, but it comes with the risk of missing some failures in the un-evaluated responses.
x??

---

#### Biases in AI Judges
Background context: AI judges can have biases similar to those of human evaluators. These biases might favor their own responses or be influenced by other factors such as the wording of the prompt. Understanding these biases helps in interpreting and potentially mitigating them during evaluation.
:p What is self-bias in AI judges?
??x
Self-bias occurs when an AI judge tends to score its own generated responses more favorably compared to those generated by other models or systems. This bias arises because the same model used for generation also scores its own outputs.

For example:
- If GPT-4 generates a response and then uses itself as an evaluator, it might give higher scores due to internal mechanisms that preferentially rate its own responses.
x??

---

#### Position Bias of AI Models
Background context explaining the concept. AI models often exhibit a preference for certain positions in response lists, such as favoring longer responses or the first position. This bias can be mitigated by repeating tests with different orderings or carefully crafted prompts.

:p What is the position bias of AI models?
??x
Position bias refers to an AI model's tendency to favor answers based on their position in a list. For instance, GPT-4 has a 10% higher win rate for itself compared to Claude-v1, which has a 25% higher win rate. This bias can be observed when comparing longer responses versus shorter ones; Saito et al. (2023) found that in creative tasks, the longer response is almost always preferred if it's significantly longer.

For example:
- If the first answer is "The quick brown fox jumps over the lazy dog" and the second answer is "The quick brown fox," a model with position bias might prefer the first one due to its length.
??x
---

#### Recency Bias in Humans
Background context explaining the concept. Humans tend to favor the last item they see, which can affect their judgment when presented with multiple options.

:p What is recency bias?
??x
Recency bias refers to a cognitive bias where humans prefer or remember information that occurs more recently. This applies particularly well to evaluations and judgments where decisions are made based on recent exposure.

For example:
- If you have been shown three answers, you might rate the last one as the best because it was most recently in your mind.
??x
---

#### Verbal Bias in AI Models
Background context explaining the concept. Some AI models prefer longer responses regardless of their quality, often due to verbosity bias.

:p What is verbal bias in AI models?
??x
Verbal bias refers to a phenomenon where AI models favor longer answers over shorter ones, even if those longer answers are factually incorrect or of lower quality. This was observed by Wu and Aji (2023) who found that both GPT-4 and Claude-v1 preferred longer responses (~100 words) with factual errors over shorter correct responses (~50 words).

For example:
```python
# Pseudocode for a simple evaluation function
def evaluate_response(response_length, factually_correct):
    if response_length > 99 or not factually_correct:
        return -1  # Penalize long incorrect answers heavily
    else:
        return 1  # Encourage short correct answers
```
??x
---

#### Evaluation Methodology: Self-Critique
Background context explaining the concept. Self-critique involves an AI model evaluating its own responses, which can help improve reliability and accuracy through sanity checks or nudging the model to revise its responses.

:p What is self-critique in AI models?
??x
Self-critique refers to a method where an AI model evaluates its own generated response. This technique helps ensure that the model's outputs are reliable and accurate, as it can catch and correct errors through internal feedback loops. 

For example:
```python
# Pseudocode for self-critique
def self_critique(response):
    # Check if the response is logically consistent with known facts or rules
    if check_consistency(response) == False:
        return "Is this answer correct? Final response: No, it's incorrect. The correct answer is [correct answer]."
    else:
        return "This response seems good."
```
??x
---

#### Evaluation Methodology: Weaker Judge vs Stronger Model
Background context explaining the concept. When evaluating a stronger model with a weaker one, there are trade-offs in terms of cost and latency. Using a weaker model can help manage resources while still ensuring that evaluations are conducted.

:p Can a weaker model be used to evaluate responses from a stronger one?
??x
Yes, a weaker model can effectively evaluate responses generated by a stronger one. This approach helps balance the use of computational resources while ensuring that high-quality evaluations are performed.

For example:
```python
# Pseudocode for evaluating with a weaker model
def evaluate_responses(strong_model_responses, weak_model):
    evaluated_responses = []
    for response in strong_model_responses:
        evaluation = weak_model.evaluate(response)
        evaluated_responses.append(evaluation)
    return evaluated_responses
```
??x
---

#### Specialized Judges vs General-Purpose Judges
Background context explaining the concept. Specialized judges are trained to make specific judgments, whereas general-purpose judges can handle a wide range of tasks but may be less reliable for specific evaluations.

:p What is the difference between specialized and general-purpose judges?
??x
Specialized judges are tailored to make specific judgments using particular criteria or scoring systems. They can provide more reliable results for specific tasks due to their focused training. In contrast, general-purpose judges are broader in scope but may not perform as well on niche evaluations.

For example:
```java
// Pseudocode for a specialized judge
public class MathJudge {
    public int evaluateResponse(String prompt, String response) {
        // Logic to evaluate math problems specifically
        if (response.equals("13")) return 5; // Correct answer is 13
        else return 0;
    }
}
```
??x
---

#### Preference Models in AI Evaluation
Background context explaining the concept. Preference models take inputs like (prompt, response1, response2) and determine which response is preferred by users based on given criteria.

:p What are preference models in the context of AI evaluation?
??x
Preference models predict user preferences between two or more responses for a given prompt. This type of model can help align AI systems with human preferences, making evaluations more accurate and easier to perform.

For example:
```python
# Pseudocode for a preference model
def determine_preference(prompt, response1, response2):
    # Logic to determine which response is preferred
    if logic_to_compare_responses(response1, response2) == "response1":
        return 1  # Response 1 is preferred
    else:
        return 0  # Response 2 is preferred
```
??x
---

