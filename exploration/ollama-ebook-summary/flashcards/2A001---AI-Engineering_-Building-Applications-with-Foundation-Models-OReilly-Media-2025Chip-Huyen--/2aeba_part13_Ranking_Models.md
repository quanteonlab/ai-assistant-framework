# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 13)

**Starting Chapter:** Ranking Models with Comparative Evaluation

---

#### Likert Scale and PandaLM Output

Background context: The use of a Likert scale to evaluate responses is mentioned. This method involves providing numerical ratings that reflect the degree of agreement with a statement or question. In this example, PandaLM generates two responses, which are evaluated using a human judge.

:p What does the Likert scale represent in this context?
??x
The Likert scale represents a method for quantifying responses to survey questions, where participants rate their level of agreement on a numerical scale. This can be used here to evaluate and compare generated responses from AI models.
x??

---

#### Pointwise Evaluation

Background context: Pointwise evaluation involves evaluating each model independently, then ranking them by their scores.

:p How does pointwise evaluation work?
??x
Pointwise evaluation evaluates each model separately and assigns a score to it. The models are then ranked based on these individual scores. For instance, in a dancing contest, each dancer is evaluated individually, given a score, and the highest-scoring dancer is chosen.
x??

---

#### Comparative Evaluation

Background context: Comparative evaluation involves comparing models against each other and computing rankings from comparison results.

:p How does comparative evaluation differ from pointwise evaluation?
??x
Comparative evaluation compares models side-by-side to determine which performs better. It ranks models based on user preference or performance in direct comparisons, whereas pointwise evaluation assigns a score to each model independently without comparing them directly.
x??

---

#### ChatGPT Example

Background context: An example of ChatGPT asking users to compare two outputs is provided.

:p What does ChatGPT do in this comparative evaluation scenario?
??x
ChatGPT asks its users to compare two outputs side by side. Users are then asked to pick the winner, which helps rank models based on user preference.
x??

---

#### Ranking Models with Comparative Evaluation

Background context: This method uses pairwise comparisons to rank models.

:p How is ranking computed in comparative evaluation?
??x
Ranking is computed by comparing pairs of models and determining the win rate for each model. The more often a model wins, the higher it ranks. For instance, if Model A beats Model B 90% of the time, Model A has a higher rank.
x??

---

#### Preference-Based Voting

Background context: Preference-based voting is discussed as a method to avoid misaligned behaviors in AI.

:p Why can preference-based voting lead to incorrect signals?
??x
Preference-based voting can lead to incorrect signals if used improperly. For example, asking users to choose between “Yes” and “No” on a factual question might yield misleading answers because the voters may not have enough knowledge about the subject.
x??

---

#### A/B Testing vs Comparative Evaluation

Background context: The difference between A/B testing and comparative evaluation is explained.

:p What is the key difference between A/B testing and comparative evaluation?
??x
In A/B testing, a user sees one model's output at a time. In comparative evaluation, users see outputs from multiple models simultaneously and are asked to compare them.
x??

---

#### Win Rate Calculation

Background context: The win rate is used to rank models based on their performance in comparisons.

:p How do you calculate the win rate of a model?
??x
The win rate of a model is calculated by counting the number of times it wins against another model and dividing that by the total number of comparisons. For example, if Model A wins 90% of its matches against Model B, the win rate for Model A in those comparisons is 90%.
x??

---

#### Rating Algorithms

Background context: Various rating algorithms like Elo, Bradley–Terry, and TrueSkill are mentioned.

:p Which algorithm might be used to rank models based on comparative evaluations?
??x
Rating algorithms such as Elo, Bradley–Terry, or TrueSkill can be used to rank models. For instance, the Bradley–Terry algorithm was used by LMSYS’s Chatbot Arena after switching from the Elo algorithm due to sensitivity issues.
x??

---

#### Model Pair Comparison Example

Background context: An example of model pair comparisons is provided in Table 3-6.

:p How would you determine the ranking of five models based on the given win rates?
??x
To rank the models, calculate their win rates across all comparisons. The model with the highest overall win rate ranks highest. For instance, if Model 1 has a high win rate against other models, it will be ranked higher than those with lower win rates.
x??

---

#### Ranking Quality

Background context: The quality of rankings is evaluated based on future match outcomes.

:p How do you assess the quality of a ranking in comparative evaluation?
??x
The quality of a ranking is assessed by how well it predicts future match outcomes. A correct ranking should indicate that higher-ranked models perform better against lower-ranked ones.
x??

---

#### Scenario Analysis for Model Evaluation
Background context explaining the need to understand different scenarios when evaluating models. Mention the example of model A and B, where one is better than the other but the reasons could vary.

:p In the given scenario, what are the possible explanations for why model B might be considered better than model A?
??x
The answer with detailed explanations:
In the provided scenario, there can be several explanations as to why model B is deemed better than model A. These include:

1. **Model B is good but Model A is bad**: This means that while both models have their strengths and weaknesses, B outperforms A due to its overall quality.
2. **Both Models Are Bad but Model B Performs Slightly Better**: Both models might be inadequate in resolving tickets or providing satisfactory responses, yet model B has a slight edge over A.
3. **Both Models Are Good but Model B Performed Better in the Specific Test Case**: In some applications, even if both models are good, the specific test cases used for comparison may favor one model due to various factors.

These scenarios highlight the need for additional evaluation methods beyond comparative evaluation to determine which scenario is true.
x??

---

#### Performance Boost Uncertainty
Background context explaining the uncertainty in performance improvement based on slight changes in win rates. Include examples of how a 1% change can significantly impact some applications but not others.

:p Why does it matter that a 1% increase in win rate might not translate to a significant performance boost across all applications?
??x
The answer with detailed explanations:
A 1% increase in the win rate may lead to different levels of performance improvement depending on the application. For instance, in some cases, such a small change could result in minimal improvements or no noticeable difference at all. However, in other scenarios, this slight increase might trigger significant enhancements.

In the context of using model A for customer support, if model B wins against A 51% of the time but resolving tickets is only one aspect of customer support, the actual performance boost from switching to model B could be uncertain and may not justify the cost. This uncertainty complicates the cost-benefit analysis when deciding whether to replace A with B.

For example, consider a scenario where:
- Model A resolves 70% of all tickets.
- Model B wins against A in 51% of the cases but does not necessarily resolve more tickets overall.

This ambiguity necessitates additional evaluations beyond comparative testing to determine if the performance gain from model B justifies its higher cost.
x??

---

#### Future of Comparative Evaluation
Background context explaining the ongoing debates and benefits regarding comparative evaluation. Discuss the role of human evaluators in detecting subtle differences between models even when exact scores are hard to assign.

:p What are the main benefits of using comparative evaluation, especially with stronger AI models?
??x
The answer with detailed explanations:
Comparative evaluation has several key benefits, particularly as AI models become increasingly powerful:

1. **Ease of Comparison**: It's often easier for humans to compare two outputs rather than assigning concrete scores. This is particularly useful when models surpass human performance in certain tasks.
2. **Human Preference Capture**: Comparative evaluations help capture the qualities that matter most—human preferences—which can be difficult to quantify with benchmark tests.
3. **Resistance to Gaming**: Unlike benchmarking, comparative evaluation is harder to manipulate or game because it relies on direct comparison rather than predefined metrics.

For example, in the Llama 2 paper, even when models venture into complex writing beyond human capabilities, humans can still provide valuable feedback through comparative evaluations (Touvron et al., 2023).

These benefits make comparative evaluation a robust approach despite its limitations.
x??

---

#### Evaluating Foundation Models
Background context explaining the challenges in evaluating open-ended and powerful AI models. Highlight the use of language modeling metrics like perplexity and cross-entropy, as well as subjective metrics.

:p What are some key challenges when evaluating foundation models, and why do many teams rely on human evaluators?
??x
The answer with detailed explanations:
Evaluating foundation models presents several significant challenges:

1. **Catastrophic Failures**: Stronger AI models have a higher risk of catastrophic failures, making evaluation more critical but also more challenging.
2. **Complexity and Human Preferences**: Evaluating open-ended responses requires capturing human preferences, which can be difficult to quantify with exact metrics.

Many teams rely on human evaluators because:

- **Human Preferences**: Capturing nuanced human preferences is crucial for applications like customer support or creative tasks.
- **Benchmark Limitations**: Perfect scores in benchmarks may make them less useful as models continue to improve. Comparative evaluation remains a reliable option even when benchmarks become saturated.

However, human evaluation can be costly and time-consuming, leading many teams to explore automated methods like exact metrics (perplexity, cross-entropy) or subjective metrics (similarity scores).

These challenges underscore the importance of integrating both automatic and human evaluations into the model development process.
x??

---

#### Importance of Evaluation in AI
Background context explaining why evaluation is essential as models become stronger and more complex. Discuss the balance between exact and subjective evaluation methods.

:p Why is evaluation so critical when dealing with advanced AI models?
??x
The answer with detailed explanations:
Evaluation is crucial for several reasons, especially as AI models become more powerful:

1. **Risk Management**: Stronger models pose a higher risk of catastrophic failures, necessitating thorough testing.
2. **Quality Assurance**: Ensuring that the model meets the intended quality and performance standards is vital.

Balancing exact evaluation methods (like perplexity and cross-entropy) with subjective evaluations (similarity scores, AI judges) provides a more comprehensive assessment:

- **Exact Metrics**: Provide quantitative insights but may not capture all aspects of human preference.
- **Subjective Metrics**: Focus on qualitative aspects but are highly dependent on the judge's perspective.

The ideal approach often involves combining these methods to leverage their strengths and mitigate weaknesses. This integration helps in building reliable evaluation pipelines for open-ended applications.
x??

---

#### AI Judges and Comparative Evaluation
Background context explaining the role of AI judges in comparative evaluation, especially with foundation models. Discuss the limitations and potential uses of AI judges.

:p How do AI judges fit into the evaluation process, particularly with advanced language models like Llama 2?
??x
The answer with detailed explanations:
AI judges play a crucial role in evaluating open-ended responses from advanced language models:

1. **Comparative Evaluation**: They help in comparing model outputs based on human preferences.
2. **Preference Prediction**: AI judges predict which response users prefer, providing subjective but valuable insights.

However, AI judges have limitations:

- **Reliability Issues**: Their judgments can change over time, making them unreliable benchmarks for tracking application changes.
- **Interpretation Challenges**: Scores from different judges may not be directly comparable due to their subjective nature.

Despite these challenges, AI judges are useful tools when combined with exact metrics and human evaluations. They help in capturing subtle differences that might be missed by purely objective methods.

For example, comparing two responses generated by Llama 2 can provide valuable insights even if the models perform well on standard benchmarks.
x??

---

#### Cost-Benefit Analysis for Model Swaps
Background context explaining how comparative evaluation alone is insufficient to make decisions about model swaps due to uncertainties in performance gains and costs involved.

:p Why is it difficult to justify swapping one model for another based solely on a 1% increase in win rate?
??x
The answer with detailed explanations:
Swapping one model for another based solely on a small increase in win rate (e.g., from 70% to 70.5%) requires more than just comparative evaluation:

- **Performance Uncertainty**: A 1% change might not translate into significant performance improvements across all aspects of the application.
- **Cost Considerations**: If the new model costs twice as much, the cost-benefit analysis must consider both financial and operational factors.

For example, if a 51% win rate with Model B translates to resolving fewer tickets overall than Model A but requires double the cost, the decision to switch models becomes complex. Additional evaluations are needed to determine whether the performance boost justifies the higher costs.

Therefore, comparative evaluation must be supplemented by other forms of assessment to make informed decisions.
x??

---

#### Evaluation Pipelines for Open-Ended Applications
Background context explaining the need for robust evaluation pipelines specifically tailored for open-ended applications. Highlight the integration of various evaluation methods.

:p How can we build reliable evaluation pipelines for open-ended applications, and why is this important?
??x
The answer with detailed explanations:
Building reliable evaluation pipelines for open-ended applications involves integrating multiple evaluation methods:

1. **Exact Metrics**: Provide quantitative insights like perplexity and cross-entropy.
2. **Subjective Metrics**: Capture qualitative aspects through AI judges or human evaluators.
3. **Comparative Evaluation**: Directly compare model outputs based on performance.

These combined approaches help in addressing the limitations of any single method, ensuring a comprehensive evaluation process:

```java
public class EvaluationPipeline {
    private List<ExactMetricEvaluator> exactEvaluators;
    private List<SubjectiveMetricEvaluator> subjectiveEvaluators;
    private ComparativeEvaluator comparativeEvaluator;

    public void evaluateModels(List<Model> models) {
        // Evaluate using exact metrics
        for (Model model : models) {
            model.evaluate(exactEvaluators);
        }

        // Evaluate using subjective metrics
        for (Model model : models) {
            model.evaluate(subjectiveEvaluators);
        }

        // Compare models based on comparative evaluation
        comparativeEvaluator.compare(models);
    }
}
```

This pipeline ensures that various aspects of the models are thoroughly evaluated, providing a balanced and reliable assessment.

The importance lies in ensuring that advanced AI applications meet high standards of performance and quality.
x??

---

#### Evaluation Criteria for AI Applications
Background context: This section discusses the importance of evaluating AI applications to ensure they meet their intended purposes. It highlights common issues such as ensuring factual consistency and domain-specific capabilities are measured accurately.

:p What criteria should be used to evaluate AI applications?
??x
Evaluating AI applications involves several key criteria, including:
- Factual Consistency: Ensuring the model provides accurate information.
- Domain-Specific Capabilities: Measuring abilities in specific domains like math, science, reasoning, and summarization.
- User Feedback: Assessing user satisfaction with features.
- Performance Metrics: Using appropriate metrics to evaluate performance.

For example, for factual consistency, you might use techniques such as fact-checking APIs or human evaluators. For domain-specific capabilities, benchmarks can be used to measure accuracy in relevant tasks.

```java
public class EvaluationCriteria {
    public static void checkFactualConsistency(String[] facts) {
        // Check each fact against a trusted database
        for (String fact : facts) {
            if (!verifyFact(fact)) {
                System.out.println("Inconsistent: " + fact);
            }
        }
    }

    private static boolean verifyFact(String fact) {
        // Dummy implementation to check fact consistency
        return true;
    }
}
```
x??

---

#### Model Selection for AI Applications
Background context: With a growing number of foundation models, choosing the right model for an application can be overwhelming. This section discusses how to select the appropriate model based on benchmarks and public leaderboards.

:p How do you choose the right model for your application?
??x
Choosing the right model involves several steps:
1. **Identify Evaluation Criteria**: Define what metrics are important (e.g., accuracy, speed, robustness).
2. **Review Benchmarks**: Use established benchmarks like GLUE, SuperGLUE, or others relevant to your domain.
3. **Consider Public Leaderboards**: Platforms like Hugging Face Model Hub or ML Competitions can provide aggregate scores and rankings.
4. **Evaluate Proprietary vs Open Source Models**: Consider the costs and benefits of hosting models internally versus using model APIs.

For example, if you need a model for natural language understanding, you might use the SuperGLUE benchmark to compare different models.

```java
public class ModelSelection {
    public static String chooseModel(List<String> benchmarks) {
        // Dummy implementation to select based on evaluation criteria
        return "model1";
    }
}
```
x??

---

#### Developing an Evaluation Pipeline
Background context: An effective evaluation pipeline helps guide the development of AI applications over time. It integrates various techniques learned throughout the book to evaluate specific applications.

:p What is an evaluation pipeline for AI applications?
??x
An evaluation pipeline is a systematic approach to continuously assess and improve AI models in real-world scenarios. It typically includes:
1. **Initial Evaluation**: Conducting initial testing using defined criteria.
2. **Continuous Monitoring**: Regularly assessing model performance post-deployment.
3. **Feedback Loop**: Incorporating user feedback and performance data into model improvements.

For example, a pipeline might involve setting up A/B tests to compare new vs old models and collecting user feedback through surveys or direct interactions.

```java
public class EvaluationPipeline {
    public static void setupPipeline() {
        // Step 1: Initial evaluation
        evaluateModel();

        // Step 2: Continuous monitoring
        monitorPerformance();

        // Step 3: Feedback loop
        collectUserFeedback();
    }

    private static void evaluateModel() {
        // Dummy implementation to simulate model evaluation
    }

    private static void monitorPerformance() {
        // Dummy implementation to monitor performance over time
    }

    private static void collectUserFeedback() {
        // Dummy implementation to gather user feedback
    }
}
```
x??

---

#### Example of A/B Testing for AI Applications
Background context: A/B testing is crucial in evaluating the impact of AI applications. This section discusses why it's important and provides a framework for conducting such tests.

:p Why is A/B testing important for AI applications?
??x
A/B testing is essential because:
- It helps differentiate between the actual impact of an application and other factors (e.g., promotional campaigns, new product launches).
- It ensures that improvements are driven by the application itself rather than external variables.
- It provides quantitative data to measure the effectiveness of changes.

For example, if a used car dealership wants to test a model predicting car values, they might run an A/B test where one group sees predictions and another doesn't. The results can then be compared to determine the model's value.

```java
public class ABTesting {
    public static void conductTest() {
        // Dummy implementation for A/B testing setup
        System.out.println("A/B Test Setup Complete");
    }
}
```
x??

#### Evaluation-Driven Development
Evaluation-driven development is an approach where evaluation criteria are defined before building an AI application. This ensures that the application's value can be measured, aligning business decisions with return on investment (ROI). This method is inspired by test-driven development in software engineering but adapted for AI applications.
:p What is the core idea of evaluation-driven development?
??x
The core idea of evaluation-driven development involves defining clear evaluation criteria before building an AI application to ensure it demonstrates value. This approach helps align business decisions with ROI, making sure that developed applications can be measured and justified in a business context.

---
#### Common Enterprise Applications with Clear Evaluation Criteria
Examples include recommender systems, fraud detection systems, and generative AI use cases like coding. These applications are commonly deployed because their success can be easily quantified.
:p What are some common enterprise applications that benefit from evaluation-driven development?
??x
Some common enterprise applications that benefit from evaluation-driven development are:
- **Recommender Systems**: Measured by engagement or purchase-through rates.
- **Fraud Detection Systems**: Measured by the amount of money saved from prevented frauds.
- **Coding Applications**: Measured using functional correctness since generated code can be evaluated.

---
#### Domain-Specific Capability
This capability metric evaluates how well a model understands specific domains, such as legal contracts. For instance, summarizing a legal contract requires understanding its nuances.
:p How does domain-specific capability apply to AI models?
??x
Domain-specific capability measures how well an AI model understands and processes content from a particular field or domain. For example, summarizing a legal contract involves evaluating the model's ability to grasp complex terminology and structures specific to legal documents.

---
#### Generation Capability
This metric assesses the coherence and faithfulness of generated outputs, such as summaries or translations. It ensures that the output is not only correct but also makes sense in context.
:p How does generation capability impact AI applications?
??x
Generation capability evaluates how coherent and faithful the generated content is. For instance, when summarizing a document, it checks if the summary retains key information accurately while being concise and understandable.

---
#### Instruction-Following Capability
This metric ensures that an AI model adheres to specific instructions or constraints, like formatting requirements or length limits.
:p How does instruction-following capability ensure AI application quality?
??x
Instruction-following capability ensures that generated content meets specified requirements. For example, if you ask a model to summarize a document and specify the desired format or length, this metric checks whether the output complies with those instructions.

---
#### Cost and Latency Metrics
These metrics measure the economic impact of using an AI application by evaluating its operational costs and response times.
:p How do cost and latency metrics affect AI applications?
??x
Cost and latency metrics assess how much an application will cost to use and how quickly it can provide results. For instance, a model that generates summaries needs to be evaluated on both the financial impact of usage and the time taken for responses.

---
#### Example: Evaluating Coding Agents
When building a coding agent, you need to evaluate its ability to write code based on specific criteria.
:p How would you evaluate a coding agent?
??x
Evaluating a coding agent involves checking if it can generate functional, coherent, and correct code. Specific metrics could include:
- **Functional Correctness**: Does the generated code compile and run without errors?
- **Code Coherence**: Is the code well-written and follows best practices?

```java
// Example of evaluating functional correctness
public boolean testFunctionality(String generatedCode) {
    try {
        // Compile and run generated code
        Class<?> compiledClass = Class.forName("GeneratedClass");
        Method method = compiledClass.getMethod("exampleMethod");
        method.invoke(null);
        return true;
    } catch (Exception e) {
        return false;
    }
}
```
x??
The example provided is a simple function to test if generated code compiles and runs without errors. This demonstrates the concept of functional correctness in evaluating coding agents.

---

#### Domain-Specific Capabilities for AI Applications
Background context: To build an application that translates from Latin to English, you need a model that understands both languages. The model's capabilities are constrained by its configuration (like architecture and size) and training data. If it never saw Latin during training, it won't understand it.
:p What criteria should be considered when evaluating the domain-specific capabilities of a translation model?
??x
When evaluating the translation model, consider exact evaluation for domain-specific tasks like translating Latin to English. This involves checking if the translated text is accurate and coherent in both languages.
```python
def evaluate_translation(model_output, reference):
    # Logic to compare model_output with reference
    pass
```
x??

---

#### Evaluation Criteria: Exact vs. Functional Correctness
Background context: The evaluation criteria differ based on the type of task. For coding-related tasks like SQL generation, functional correctness is crucial. However, for non-coding domains, exact evaluations are common.
:p What types of tasks use functional correctness as a primary evaluation metric?
??x
Coding-related tasks typically use functional correctness as their primary evaluation metric. This means checking if the generated code runs without errors and produces the expected output.
```python
def evaluate_functional_correctness(generated_code, test_cases):
    # Logic to execute generated_code with test_cases and check for correctness
    pass
```
x??

---

#### Efficiency and Cost in Evaluating AI Models
Background context: When evaluating models, efficiency and cost can be critical. Just like a car that consumes too much fuel, an SQL query that takes too long or uses excessive memory might not be usable.
:p How does the BIRD-SQL benchmark measure both execution accuracy and efficiency?
??x
BIRD-SQL measures both execution accuracy and efficiency by comparing the runtime of the generated SQL queries with those of ground truth queries. This ensures that models produce correct results while also being efficient.
```python
def evaluate_bird_sql(ground_truth_query, generated_query):
    # Measure runtime for both queries and compare
    pass
```
x??

---

#### Close-Ended vs. Open-Ended Evaluation Tasks
Background context: Most public benchmarks follow a close-ended approach for non-coding domain capabilities, which involves multiple-choice questions or similar tasks that are easier to verify.
:p Why might an open-ended task be unsuitable in a benchmark like AGIEval?
??x
AGIEval excludes open-ended tasks to avoid inconsistent assessments. Open-ended tasks can lead to subjective grading, making it difficult to maintain consistency across different evaluators. Close-ended tasks ensure more reliable and reproducible results.
```python
def evaluate_close_ended(question_options, expected_answer):
    # Logic to select the correct answer from question_options
    pass
```
x??

---

#### Multiple-Choice Questions in Benchmarks
Background context: In April 2024, 75 percent of tasks in Eleuther's lm-evaluation-harness are multiple-choice. Examples include UC Berkeley’s MMLU (2020), Microsoft’s AGIEval (2023), and the AI2 Reasoning Challenge (ARC-C) (2018).
:p Provide an example question from the MMLU benchmark.
??x
Example question from the MMLU benchmark:
Question: One of the reasons that the government discourages and regulates monopolies is that
(A) Producer surplus is lost and consumer surplus is gained.  
(B) Monopoly prices ensure productive efficiency but cost society allocative efficiency.  
(C) Monopoly firms do not engage in significant research and development.  
(D) Consumer surplus is lost with higher prices and lower levels of output.
Label: (D)
x??

---

