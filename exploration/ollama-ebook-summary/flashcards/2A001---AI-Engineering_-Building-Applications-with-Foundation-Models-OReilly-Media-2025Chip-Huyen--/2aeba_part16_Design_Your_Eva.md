# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 16)

**Starting Chapter:** Design Your Evaluation Pipeline. Step 1. Evaluate All Components in a System

---

#### Evaluating Open-Ended Tasks
Open-ended tasks are more complex to evaluate compared to close-ended tasks. In open-ended tasks, there isn't a straightforward correct or incorrect answer; instead, the quality of the output depends on its relevance and usefulness.
:p How do you evaluate an AI application in an open-ended task?
??x
In evaluating open-ended tasks, you need to consider multiple factors such as the accuracy, relevance, and usefulness of the generated outputs. One effective method is to use a combination of automated metrics (like BLEU score for text generation) and human judgment to assess the quality of the output.
```python
def evaluate_open_ended_task(output, reference):
    # Example metric: BLEU score for text similarity
    bleu_score = calculate_bleu_score(output, reference)
    
    # Human evaluation input
    user_feedback = get_user_feedback(output)

    return {"bleu_score": bleu_score, "user_feedback": user_feedback}
```
x??

---

#### Evaluating Components Independently
Evaluating the components of a complex AI application independently helps in pinpointing specific areas where the system might be failing. This is especially useful when multiple steps contribute to the final output.
:p Why should you evaluate each component of a complex system separately?
??x
Evaluating each component separately ensures that issues can be isolated and addressed more effectively. For example, if your application consists of two steps: extracting text from a PDF and then identifying an employer, evaluating these components independently allows you to determine whether the issue lies in the text extraction or the employer identification.
```python
def evaluate_pdf_extraction(pdf_text, ground_truth):
    similarity_score = calculate_similarity(pdf_text, ground_truth)
    return similarity_score

def evaluate_employer_identification(extracted_text, employer_name):
    accuracy = calculate_accuracy(extracted_text, employer_name)
    return accuracy
```
x??

---

#### Turn-Based Evaluation vs. Task-Based Evaluation
Turn-based evaluation focuses on the quality of individual outputs in a conversation or interaction, whereas task-based evaluation assesses whether the system can complete an entire task.
:p What are the differences between turn-based and task-based evaluations?
??x
In turn-based evaluation, you evaluate the quality of each output independently. For example, if your application is a chatbot that debugs Python code, evaluating each response individually helps in understanding how well it asks for information or provides advice.

Task-based evaluation evaluates whether the system can complete an entire task successfully. It considers the overall performance and efficiency (e.g., number of turns) to solve the problem.
```python
def evaluate_turn_quality(response, expected_response):
    similarity_score = calculate_similarity(response, expected_response)
    return similarity_score

def evaluate_task_completion(successful, total_turns):
    score = {
        "successful": successful,
        "turns": total_turns
    }
    return score
```
x??

---

#### Twenty-Questions Benchmark as Task-Based Evaluation
The Twenty-Questions benchmark is a task-based evaluation method inspired by the classic game. It measures how well an AI can identify a concept through a series of yes/no questions.
:p How does the Twenty-Questions benchmark work?
??x
In the Twenty-Questions benchmark, one instance of the model (Alice) chooses a concept from a predefined set. Another instance of the model (Bob) asks a series of yes/no questions to try to identify the chosen concept. The evaluation is based on whether Bob correctly identifies the concept and how many questions it takes.
```python
def twenty_questions_benchmark(concept, questions):
    correct_identification = 0
    for question in questions:
        if question["answer"] == "yes":
            # Process yes answer
            pass
        else:
            # Process no answer
            pass
        
        if concept in question["concept"]:
            correct_identification += 1
    
    score = {
        "correct": correct_identification,
        "total_questions": len(questions)
    }
    return score
```
x??

---

#### Importance of Clear Evaluation Guidelines
Background context: A clear evaluation guideline is crucial for ensuring consistent and reliable evaluation of AI systems. Ambiguous guidelines can lead to misleading scores, making it difficult to identify bad responses effectively.

:p Why are clear evaluation guidelines important?
??x
Clear evaluation guidelines are essential because they define both what the application should do and what it shouldn't do. This clarity helps in identifying good versus bad responses accurately. Without a well-defined guideline, evaluations can be subjective and inconsistent, leading to unreliable results. For example, an AI chatbot designed for customer support must not answer unrelated questions such as those about upcoming elections.
x??

---

#### Defining Evaluation Criteria
Background context: Defining evaluation criteria is crucial because good responses are often more nuanced than simply being correct. Multiple types of feedback can be used to evaluate applications comprehensively.

:p What should you consider when defining evaluation criteria?
??x
When defining evaluation criteria, it's important to think about the specific context and goals of your application. For instance, for a customer support chatbot, key criteria might include relevance, factual consistency, and safety. These criteria help in determining whether a response is appropriate and useful.

For example:
- **Relevance**: The response directly addresses the user’s query.
- **Factual Consistency**: The response aligns with known facts or context.
- **Safety**: The response does not contain harmful content.

You can use real user queries to test these criteria by generating multiple responses and evaluating their quality based on these defined standards.
x??

---

#### Creating Scoring Rubrics
Background context: Once you have defined your evaluation criteria, creating a scoring rubric helps in systematically assessing the quality of responses. This involves choosing a scoring system and providing examples for different scores.

:p How should you create a scoring rubric?
??x
To create a scoring rubric, follow these steps:
1. **Choose a Scoring System**: Decide on the type of scale (e.g., binary, 1-5, between 0 and 1).
2. **Define Examples for Each Score**: For each criterion, determine what responses would receive different scores.
3. **Validate with Humans**: Ensure that the rubric is clear and understandable by testing it with coworkers or friends.

Example: To evaluate factual consistency:
- Binary system: 0 (inconsistent) vs. 1 (consistent)
- Tri-scale system: -1 (contradictory), 0 (neutral), 1 (entailment)

For instance, a response that perfectly aligns with the context would get a score of 1 in both systems.

```java
public class ScoringRubric {
    public static int evaluateConsistency(String response, String context) {
        // Logic to determine consistency and return a score from -1 to 1
        if (isConsistent(response, context)) {
            return 1;
        } else if (isContradictory(response, context)) {
            return -1;
        }
        return 0; // Neutral response
    }

    private static boolean isConsistent(String response, String context) {
        // Implementation logic to check consistency
        return true; // Placeholder implementation
    }

    private static boolean isContradictory(String response, String context) {
        // Implementation logic to check contradiction
        return false; // Placeholder implementation
    }
}
```
x??

---

#### Tying Evaluation Metrics to Business Goals
Background context: Evaluating AI systems should ultimately align with business goals. Understanding the impact of evaluation metrics on these goals helps in making informed decisions about resource allocation and improvement strategies.

:p How can you tie evaluation metrics to business metrics?
??x
Tying evaluation metrics to business metrics involves understanding how specific metrics like factual consistency translate into tangible business outcomes. For instance, a chatbot's performance might be evaluated based on its ability to automate customer support requests efficiently.

Example:
- Factual consistency of 80%: Automate 30% of customer support requests.
- Factual consistency of 90%: Automate 50%.
- Factual consistency of 98%: Automate 90%.

This mapping helps in planning and prioritizing improvements based on the expected business impact. For example, improving factual consistency to 90% could increase automation by 20%, which might be a valuable investment if it leads to significant cost savings or customer satisfaction.

```java
public class BusinessMetricsMapper {
    public static int getAutomationPercentage(double factualConsistency) {
        // Mapping factual consistency to automation percentage
        if (factualConsistency >= 98) {
            return 90;
        } else if (factualConsistency >= 90) {
            return 50;
        } else if (factualConsistency >= 80) {
            return 30;
        }
        return 0; // Default value
    }
}
```
x??

---

#### Evaluation Methods and Data Selection
Background context: After defining criteria and scoring rubrics, it's essential to choose appropriate evaluation methods and data. Different criteria may require different evaluation techniques, such as specialized classifiers for toxicity detection or semantic similarity measures.

:p What are some examples of evaluation methods that might be used in an application?
??x
Some evaluation methods include using a small, specialized toxicity classifier for toxicity detection, a semantic similarity measure to gauge the relevance between responses and questions, and an AI judge to assess factual consistency. For instance:

```java
// Example of a simple classifier usage (pseudocode)
if (toxicityClassifier.isToxic(response)) {
    score -= 1;
}
```
x??

---

#### Different Evaluation Methods for Same Criteria
Background context: Not all evaluation methods are created equal, and sometimes it's beneficial to use different methods to balance cost-effectiveness and quality. For example, a cheap classifier can be used on most data points, while an expensive AI judge is used on a small subset.

:p How might you mix evaluation methods for the same criteria?
??x
You could have a simple classifier that provides low-quality signals on 100% of your dataset, and use an expensive AI judge to provide high-quality signals on only 1% of the data. This approach allows you to maintain a certain level of confidence in your application while keeping costs manageable.

```java
// Pseudocode for mixed evaluation methods
if (cheapClassifier.isSignalStrong(response)) {
    score -= 0.5;
} else if (expensiveAIJudge.isSignalStrong(response)) {
    score -= 1;
}
```
x??

---

#### Using Logprobs for Evaluation
Background context: Logprobs can be particularly useful in classification tasks, providing insights into the model's confidence levels. This information is valuable for assessing factors like fluency and factual consistency.

:p How do logprobs help in evaluating AI systems?
??x
Logprobs provide a measure of how confident a model is about its predictions. For example:
- If the model’s probabilities are uniformly low (e.g., between 30% and 40%), it indicates uncertainty.
- High probability values (e.g., 95%) indicate high confidence.

```java
// Pseudocode for using logprobs in evaluation
if (model.logprob(response) > 80) {
    score += 1; // Model is highly confident
} else if (model.logprob(response) < 30) {
    score -= 2; // Model is uncertain
}
```
x??

---

#### Automated vs. Human Evaluation
Background context: While automatic metrics are preferred, human evaluation remains crucial, especially for open-ended responses. Human experts can provide insights that automated systems might miss.

:p When should you use human evaluation in the development process?
??x
Human evaluation should be used during both experimentation and production phases. During experimentation, reference data is often available to compare against. In production, user feedback becomes more critical since actual users are interacting with the system.

```java
// Example of integrating human evaluation (pseudocode)
if (dailyUserFeedback.containsIssues()) {
    score -= 1;
} else if (humanExperts.detectPerformanceChanges()) {
    score += 2;
}
```
x??

---

#### Annotating Evaluation Data
Background context: Curating annotated examples is essential for evaluating the performance of each system component and criterion. This process ensures that both turn-based and task-based evaluations are comprehensive.

:p Why is it important to curate a set of annotated examples?
??x
Curating annotated examples is crucial because it allows you to evaluate the performance of your application's components against known outcomes. This helps in identifying areas for improvement and ensuring that all criteria are met accurately.

```java
// Example of annotating evaluation data (pseudocode)
AnnotatedExample example = new AnnotatedExample(input, expectedOutput);
evaluationSystem.evaluate(example);
```
x??

---

#### Slice-Based Evaluation
Slice-based evaluation involves separating your data into subsets and analyzing how your system performs on each subset. This method helps in gaining a finer-grained understanding of your machine learning model's performance across different data characteristics.

:p What is slice-based evaluation, and why is it important?
??x
Slice-based evaluation is a technique where the dataset is divided into smaller, more specific subsets based on various attributes (such as user groups, input length, or usage patterns). This method helps identify potential biases, debugging issues, finding areas for improvement, and avoiding pitfalls like Simpson’s paradox. It ensures that your model performs well across different segments of the data.

For example, if you are developing a chatbot, slicing by user demographics (e.g., age, location) can help ensure the bot performs equally well for all groups.
x??

---

#### Avoiding Potential Biases
By slicing the dataset, we can identify and mitigate biases that might affect specific subsets of users. For instance, minority groups could be underrepresented or misrepresented.

:p How does slice-based evaluation help in avoiding potential biases?
??x
Slice-based evaluation helps in identifying and mitigating biases by analyzing the model’s performance across different user groups or data slices. By ensuring fair representation and performance across these slices, you can prevent biased outcomes that might disproportionately affect certain demographic groups.

For instance, if your application is a recommendation system, slicing by age group could reveal that older users are not receiving relevant recommendations as much as younger users.
x??

---

#### Debugging Performance Issues
Analyzing the model's performance on specific subsets of data can help pinpoint why it performs poorly in certain scenarios. This could be due to unique characteristics of those subsets.

:p How can slice-based evaluation aid debugging?
??x
Slice-based evaluation aids debugging by isolating and analyzing how different subsets of your data affect the model’s performance. If you notice poor performance on a specific subset, such as long inputs or text with certain topics, this indicates that these specific attributes might be problematic for your current model.

For example, if your text classification model performs poorly on longer input texts, slicing by length can help identify why and potentially lead to improvements in handling long-form data.
x??

---

#### Finding Areas for Improvement
Identifying areas where the application underperforms allows for targeted enhancements. For instance, if the system struggles with long inputs, alternative processing techniques or more suitable models might be needed.

:p How does slice-based evaluation help find areas of improvement?
??x
Slice-based evaluation helps identify specific areas that need improvement by showing where the model performs poorly across different data slices. If you observe underperformance on certain types of input (e.g., long text, complex sentences), this can guide the development of targeted enhancements.

For instance, if your application frequently makes mistakes with long inputs, slicing by length can reveal patterns and suggest that a different preprocessing technique or longer model might be necessary.
x??

---

#### Avoiding Simpson’s Paradox
Simpson's paradox occurs when aggregated data shows one trend but breaking it down into smaller groups reveals the opposite. This can mislead you if not properly accounted for.

:p What is Simpson's paradox, and how does slice-based evaluation help in avoiding it?
??x
Simpson's paradox happens when a trend that appears in different groups of data reverses when these groups are combined. Slice-based evaluation helps avoid this by analyzing the model’s performance on smaller subsets to ensure no group is being unfairly disadvantaged.

For example, if Model A performs better than Model B on overall aggregated data but worse in each subgroup, slicing can reveal why and help address the underlying issues.
x??

---

#### Multiple Evaluation Sets
Having multiple evaluation sets allows for a more comprehensive test of your model across different scenarios. This includes real production data and subsets with known error patterns.

:p How many evaluation sets should you have, and what are they used for?
??x
You should have multiple evaluation sets to represent different data slices, including the actual distribution of production data and subsets where known errors occur. These sets help ensure your model performs well in various scenarios:

- Production-like data: Represents real-world usage.
- Error-prone examples: Includes cases where the system frequently makes mistakes.
- Out-of-scope examples: Ensures the application handles unexpected inputs appropriately.

For instance, if you have an evaluation set of 100 examples, creating bootstrapped samples can help determine if your results are reliable. This ensures that different subsets yield similar performance metrics, indicating a trustworthy evaluation pipeline.
x??

---

#### Bootstrapping for Reliability
Bootstrapping involves repeatedly sampling from the dataset to create multiple evaluation sets, which helps in assessing the reliability of the model’s performance.

:p How does bootstrapping help ensure the reliability of your evaluation results?
??x
Bootstrapping enhances reliability by creating multiple samples from your original evaluation set. This process allows you to see if similar results are consistently obtained across different subsets, ensuring that your evaluation pipeline is trustworthy.

For example, if you draw 100 samples with replacement from an initial evaluation set of 100 examples and evaluate the model on each sample, consistent performance metrics indicate a reliable evaluation.

```java
// Pseudocode for bootstrapping
public class Bootstrapper {
    private List<Integer> originalSet = new ArrayList<>(Collections.nCopies(100, 1));
    
    public void bootstrap(int sampleSize) {
        List<Integer> samples = new ArrayList<>();
        
        while (samples.size() < sampleSize) {
            int index = new Random().nextInt(originalSet.size());
            samples.add(originalSet.get(index));
        }
        
        // Evaluate model on these samples
    }
}
```
x??

#### Sample Size for Statistical Significance

Background context: Determining the sample size needed to ensure statistical significance when comparing two systems, such as prompts or models, is crucial. This ensures that differences observed are not due to chance but reflect true performance variations.

:p How many samples are needed if you want to be 95% confident that a new prompt with a 10 percent higher score than the old prompt is indeed better?

??x
To determine this, we can use OpenAI's rough estimation: for every 3× decrease in score difference, the number of samples needed increases 10×.

For a 10% score improvement:
- A 30% score difference requires ~10 samples.
- A 10% score difference (which is one-third of 30%) would require 10 × 10 = 100 samples to be 95% confident that the new prompt is indeed better.

In practice, if you need even higher confidence or have a smaller initial improvement, you may need more samples. For instance, for a 3% score difference (one-tenth of 10%), you would need ~1,000 samples.
x??

---

#### Evaluation Pipeline Reliability

Background context: Ensuring that your evaluation pipeline is reliable and consistent is essential, especially when dealing with subjective evaluations like AI as a judge. This involves checking whether better responses receive higher scores and if improved metrics lead to better business outcomes.

:p How can you verify the reliability of an evaluation pipeline?

??x
To verify the reliability of an evaluation pipeline:
- Check if better responses indeed get higher scores.
- Determine if better evaluation metrics correlate with better business outcomes.
- Ensure that running the same pipeline twice yields consistent results.
- Run the pipeline multiple times with different datasets and measure the variance in the evaluation results.

You should aim to increase reproducibility and reduce variability by maintaining consistent configurations, such as setting an AI judge's temperature to 0.
x??

---

#### Evaluation Benchmarks

Background context: Understanding typical evaluation benchmarks can help you gauge the size of your evaluation sets. This is particularly useful for ensuring that your evaluations are reliable and meaningful.

:p What does Eleuther's lm-evaluation-harness suggest about the number of examples in evaluation benchmarks?

??x
According to Eleuther's lm-evaluation-harness, the median number of examples is 1,000, while the average is 2,159. This gives you an idea of how large your evaluation sets should be.

For instance:
- A typical benchmark might have around 1,000 examples.
- If you need higher confidence or more precision, consider aiming for at least 2,159 examples per evaluation run.

These benchmarks help ensure that the evaluations are robust and can detect meaningful differences between models or prompts.
x??

---

#### Inverse Scaling Prize Recommendations

Background context: The Inverse Scaling prize suggests guidelines on the minimum number of examples needed to be confident in system comparisons. These recommendations provide a practical approach to setting up evaluation pipelines.

:p What did the organizers of the Inverse Scaling prize recommend as the minimum and preferred number of evaluation samples?

??x
The organizers of the Inverse Scaling prize recommended:
- 300 examples as an absolute minimum.
- At least 1,000 examples if the data is synthesized (McKenzie et al., 2023).

These recommendations are based on ensuring that evaluations are reliable and can confidently distinguish between different system performances.

For example, if you're using synthesized data or need high confidence in your evaluation results, it's recommended to aim for at least 1,000 examples.
x??

---

#### Sample Size Estimation Formula

Background context: Determining the sample size needed to achieve a certain level of confidence can be complex but is crucial for accurate evaluations. Statistical methods provide formulas and guidelines for this.

:p How does the formula for determining the sample size work?

??x
The formula for determining the sample size (n) required for a 95% confidence interval with a specific margin of error (E) depends on the standard deviation (σ). The basic formula is:

\[ n = \left(\frac{z_{\alpha/2} \times \sigma}{E}\right)^2 \]

Where:
- \( z_{\alpha/2} \) is the Z-value corresponding to a 95% confidence level (1.96).
- σ is the standard deviation of the population.
- E is the desired margin of error.

However, for simpler and more practical scenarios, such as comparing prompts or models with known score distributions, OpenAI's rough estimation can be used:

For every 3× decrease in score difference, the number of samples needed increases 10×. For example:
- A 30% score difference requires ~10 samples.
- A 10% score difference (one-third of 30%) would require 10 × 10 = 100 samples.

This rule provides a quick way to estimate the sample size needed for a given score difference and confidence level without needing to know the exact standard deviation.
x??

---

