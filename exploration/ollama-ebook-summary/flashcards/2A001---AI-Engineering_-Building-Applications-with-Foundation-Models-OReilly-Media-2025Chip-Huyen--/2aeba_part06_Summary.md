# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 6)

**Starting Chapter:** Summary

---

#### AI Engineering as a Discipline

Background context: The emergence of AI engineering is driven by the availability of foundation models, which simplify and accelerate application development. This shift brings more emphasis on interfaces and integrates frontend engineering skills.

:p What is AI engineering?
??x
AI engineering refers to the discipline of building applications using foundation models, which are pre-trained models capable of a wide range of tasks without significant retraining. The focus shifts towards creating user-friendly interfaces and integrating these models into various platforms like web, desktop, mobile, chatbots, etc.
x??

---

#### Shift in AI Development Workflow

Background context: Traditionally, ML engineering involves data collection, model training, and then product development. However, with the availability of foundation models, developers can start by building products first and invest in data and models only if the product shows promise.

:p How has the workflow for AI engineering changed?
??x
The new workflow rewards fast iteration by allowing developers to start building products using pre-trained models first. This approach enables quicker feedback loops and more efficient development cycles.
x??

---

#### Importance of Different Categories in App Development

Background context: Table 1-6 highlights how the importance of different categories in app development has evolved with AI engineering, including less emphasis on traditional ML interfaces but more focus on prompt engineering and evaluation.

:p How does the table reflect changes in app development for AI engineering?
??x
Table 1-6 shows that while building with traditional ML is becoming less important, there is a greater focus on creating effective user interfaces (web, desktop, mobile apps), browser extensions, chatbots, and integrating AI into existing products via APIs. Prompt engineering and evaluation also become more critical due to the ease of using pre-trained models.
x??

---

#### The Rise of Full-Stack Engineers in AI

Background context: Full-stack engineers have an advantage over traditional ML engineers because they can quickly prototype ideas, gather feedback, and iterate faster.

:p Why are full-stack engineers advantageous in AI engineering?
??x
Full-stack engineers are advantageous because they possess both frontend and backend skills, allowing them to rapidly develop prototypes, collect user feedback, and iterate on their ideas more efficiently than traditional ML engineers who often focus solely on model development.
x??

---

#### New Workflow for AI Engineering

Background context: With the rise of foundation models, it is possible to start with building products first and only invest in data and models if the product shows promise. This contrasts with the traditional workflow where data collection and training come before product development.

:p How does the new AI engineering workflow differ from traditional ML engineering?
??x
The new AI engineering workflow emphasizes starting with product design and development, leveraging pre-trained foundation models to quickly build prototypes. The focus shifts towards rapid iteration based on user feedback rather than spending initial resources on data collection and model training.
x??

---

#### Overview of the Chapter

Background context: This chapter aims to explain the emergence of AI engineering as a discipline due to foundation models and provide an overview of building applications using these models.

:p What is the primary goal of this chapter?
??x
The primary goal of this chapter is to introduce AI engineering, discuss its evolution from traditional ML engineering, and outline the process needed to build applications on top of foundation models.
x??

---

#### Rapid Evolution of AI

Background context: The chapter highlights the transition from language models to large language models and how these advancements have led to new application patterns.

:p What are some notable transformations discussed in the chapter?
??x
The chapter discusses the evolution from language models to large language models, driven by self-supervision techniques. It also covers how these models incorporate other data modalities and give rise to AI engineering.
x??

---

#### Challenges in AI Engineering

Background context: The rapid growth of AI engineering presents new challenges but also opportunities for innovation.

:p What are some challenges mentioned in the chapter?
??x
Some challenges include keeping up with new techniques, discoveries, and constant engineering feats. However, these challenges can be managed by leveraging AI's ability to aggregate information.
x??

---

#### Training Data Distribution
Background context: Model developers curate training data, which significantly influences a model's capabilities and limitations. The distribution of this training data is crucial as it shapes how well the model performs on different types of tasks.

:p How does the distribution of training data affect a model?
??x
The distribution of training data impacts a model’s performance across various tasks. A model trained with diverse data will likely generalize better to unseen scenarios compared to one that was trained on a narrow set of examples. For instance, if a language model is primarily trained on tech-related texts, it may struggle when given financial or literary content.

```java
// Example code demonstrating how different datasets can be combined for training
public class DataMerger {
    public static Dataset mergeDatasets(Dataset dataset1, Dataset dataset2) {
        // Combine two datasets by concatenating their data arrays
        List<String> mergedData = new ArrayList<>(dataset1.getData());
        mergedData.addAll(dataset2.getData());

        return new Dataset(mergedData);
    }
}
```
x??

---

#### Transformer Architecture Dominance
Background context: The transformer architecture has been the dominant choice for foundation models due to its ability to handle long-range dependencies and parallelize computations efficiently. Understanding why it is so special requires delving into its design principles.

:p Why does the transformer architecture continue to dominate in model development?
??x
The transformer architecture continues to dominate because of its unique properties that enhance performance on complex tasks like natural language processing (NLP). Key reasons include:

1. **Self-Attention Mechanism**: Allows each position in the input sequence to attend over all other positions, enabling it to capture dependencies at different scales.
2. **Parallelization**: Computation can be parallelized easily due to its structure, significantly reducing training time and allowing larger models.

```java
// Pseudocode for a simple transformer layer
public class TransformerLayer {
    public OutputLayer applySelfAttention(InputLayer input) {
        // Apply self-attention mechanism to the input
        return new OutputLayer(selfAttention.apply(input));
    }
}
```
x??

---

#### Model Size Determination
Background context: The size of a model is often a critical decision, as it directly impacts performance and computational requirements. Larger models can capture more complex patterns but require more resources.

:p How do model developers determine the appropriate size for their foundation model?
??x
Model developers determine the appropriate size based on several factors:

1. **Task Complexity**: More complex tasks may require larger models to learn intricate patterns.
2. **Computational Resources**: Larger models demand significant computational power, so resource availability is a limiting factor.
3. **Training Data Quality and Quantity**: High-quality and large datasets often necessitate larger models for optimal performance.

```java
// Pseudocode for determining model size based on task complexity
public int determineModelSize(int numClasses, boolean isComplexTask) {
    if (isComplexTask) {
        return 2048; // Larger model for complex tasks
    } else {
        return 1024; // Smaller model for simpler tasks
    }
}
```
x??

---

#### Post-Training and Human Preferences
Background context: After pre-training, models need to be fine-tuned or post-trained to better align with human preferences. This process ensures that the model's outputs are more aligned with what humans expect.

:p What is the goal of post-training in foundation models?
??x
The goal of post-training is to align the model’s output with human preferences by refining its responses based on specific use cases or feedback mechanisms. This helps ensure that the model behaves predictably and produces results that are useful and understandable for end-users.

```java
// Pseudocode for a simple post-training loop
public void postTrainModel(Model model, Dataset trainingData) {
    // Iterate over the dataset to adjust the model's outputs
    for (Sample sample : trainingData.getSamples()) {
        Output prediction = model.predict(sample.getInput());
        Feedback feedback = getFeedbackFromUser(prediction); // Collect user feedback

        // Adjust the model based on the feedback
        model.updateWeights(feedback, prediction);
    }
}
```
x??

---

#### Sampling in Models
Background context: Sampling is a critical process where models choose outputs from all possible options. This process can significantly affect model behavior and performance, especially leading to issues like hallucinations.

:p What role does sampling play in AI models?
??x
Sampling plays a crucial role in how AI models generate outputs. It determines the selection of an output from multiple possibilities, affecting factors such as consistency, reliability, and overall performance. Poor sampling strategies can lead to incorrect or misleading predictions.

```java
// Pseudocode for a simple sampling strategy
public String sampleFromOptions(List<String> options) {
    // Choose one option randomly based on certain criteria (e.g., probabilities)
    return options.get(random.nextInt(options.size()));
}
```
x??

---

#### Training Data Quality and Sources

Background context: The quality of training data significantly affects the performance of AI models. Different sources of training data have varying degrees of reliability and relevance.

:p How does the choice of training data impact an AI model's performance?

??x
The choice of training data greatly influences how well an AI model performs on specific tasks. If the data is not representative or lacks certain features, the model may struggle with those aspects during inference. For instance, if a translation model has limited exposure to Vietnamese text in its training set, it will perform poorly when translating English to Vietnamese.

Code examples are less relevant here, but consider the following scenario where we evaluate the quality of different datasets:

```python
def evaluate_dataset_quality(data):
    # Example function to evaluate dataset based on criteria like diversity and relevance.
    quality_score = 0
    if "Vietnamese" in data:
        quality_score += 2
    if "Clickbait" not in data:
        quality_score += 1
    return quality_score

# Evaluating a hypothetical dataset
data_quality_score = evaluate_dataset_quality(common_crawl_data)
print(f"The quality score of the dataset is {data_quality_score}.")
```

x??

---

#### Common Crawl and Its Usage in Training AI Models

Background context: Common Crawl is an extensive web crawling project that collects vast amounts of data, often used as a source for training large language models. However, the data collected can be noisy and may contain unwanted content like fake news or biased information.

:p What are some issues with using Common Crawl as a primary training data source?

??x
Using Common Crawl as a primary training data source has several potential issues. The data is often unfiltered and can include low-quality, irrelevant, or harmful content such as clickbait, misinformation, propaganda, racism, and more. Additionally, the quality of the crawled websites varies widely, ranging from reputable news outlets to sketchy websites.

Code examples are less relevant here, but consider how a model might filter out certain types of data:

```python
def filter_common_crawl_data(data):
    # Example function to filter out noisy or harmful content.
    filtered_data = []
    for item in data:
        if "clickbait" not in item and "misinformation" not in item:
            filtered_data.append(item)
    return filtered_data

# Filtering a hypothetical dataset
filtered_common_crawl_data = filter_common_crawl_data(common_crawl_data)
print(f"The amount of filtered data is {len(filtered_common_crawl_data)}.")
```

x??

---

#### Model Performance and Data Relevance

Background context: The performance of an AI model depends on the relevance and quality of its training data. If a model lacks exposure to certain types of data, it may not perform well when encountering those scenarios during inference.

:p Why might a translation model trained only on animal images struggle with plant-based content?

??x
A translation model trained exclusively on animal images would be poorly equipped to handle plant-related content due to the lack of relevant training examples. The model's knowledge base is limited to what it has been exposed to, and without any exposure to plant images or related data during training, its ability to understand and translate terms associated with plants will be severely lacking.

Code examples are less relevant here, but consider a hypothetical example:

```python
def evaluate_model_performance(model, test_data):
    # Example function to assess model performance on specific tasks.
    correct_predictions = 0
    total_predictions = len(test_data)
    for item in test_data:
        if model.predict(item) == item['expected_output']:
            correct_predictions += 1
    return (correct_predictions / total_predictions) * 100

# Evaluating a translation model's performance on plant-related content
performance_score = evaluate_model_performance(translation_model, plant_based_test_data)
print(f"The model performs at {performance_score}% accuracy for plant-based content.")
```

x??

---

#### Curating Datasets for Specific Needs

Background context: To ensure that an AI model performs well on specific tasks or domains, it is crucial to curate datasets that are tailored to those needs. This involves selecting and preprocessing data that aligns with the target application.

:p How can one create a specialized dataset for training a translation model?

??x
To create a specialized dataset for training a translation model, you need to gather and preprocess text in both source and target languages, ensuring they are relevant to the specific domain or task. For example, if you want to train a model for translating legal documents from English to Spanish, you should collect and clean texts that pertain to legal language.

Code examples can help illustrate the process:

```python
def curate_dataset(source_lang, target_lang, domain):
    # Example function to curate a dataset based on specific language and domain.
    source_text = get_texts_from_domain(domain, lang=source_lang)
    target_text = get_translations_of_texts(source_text, target=target_lang)
    curated_data = list(zip(source_text, target_text))
    return curated_data

# Curating a dataset for legal documents in English to Spanish
curated_legal_dataset = curate_dataset("en", "es", "legal")
print(f"The curated dataset contains {len(curated_legal_dataset)} entries.")
```

x??

---

#### Finetuning Models on General-Purpose Models

Background context: Instead of training models from scratch, many teams choose to finetune existing general-purpose models on specific tasks or domains. This approach leverages the strengths of pre-trained models while tailoring them for new applications.

:p Why would a team prefer to finetune an existing model rather than train a new one?

??x
A team might prefer to finetune an existing model instead of training a new one because it saves time, resources, and computational power. Pre-trained models have already learned general patterns and features from large datasets, which can serve as a good starting point for specific tasks. Finetuning allows the model to adapt more quickly to new data without requiring extensive retraining.

Code examples could illustrate this process:

```python
def finetune_model(model, training_data):
    # Example function to fine-tune a pre-trained model.
    model.train(training_data)
    return model

# Finetuning an existing model on plant-related content
finetuned_model = finetune_model(general_purpose_model, plant_based_training_data)
print("Model has been successfully finetuned for plant-based content.")
```

x??

---

#### Data Quality and Model Performance

Background context: The quality of training data significantly impacts model performance. High-quality, smaller datasets can outperform large low-quality datasets. This is demonstrated by Gunasekar et al. (2023), who trained a 1.3B-parameter model on 7 billion tokens of high-quality coding data that outperformed larger models.

:p How does the quality of training data affect model performance?
??x
The quality of training data greatly influences how well a model performs, often more than the quantity of data. High-quality, smaller datasets can surpass large low-quality datasets in terms of performance.
x??

---

#### Language Dominance on the Internet

Background context: English accounts for almost half (45.88%) of internet data, making it significantly more prevalent than other languages. This dominance affects how well general-purpose models perform across different languages.

:p What percentage of internet data is attributed to English?
??x
English accounts for approximately 45.88% of the internet data.
x??

---

#### Common Crawl Dataset Analysis

Background context: The Common Crawl dataset, used extensively for training large language models (LLMs), shows that while some languages like English dominate, many others are severely underrepresented.

:p List the top five most common languages in Common Crawl according to their percentage in the dataset.
??x
The top five most common languages in Common Crawl based on their percentage in the dataset are:
1. English (45.8786%)
2. Russian (5.9692%)
3. German (5.8811%)
4. Chinese (4.8747%)
5. Japanese (4.7884%)
x??

---

#### Under-Represented Languages

Background context: Many languages, despite having a large number of speakers, are underrepresented in the Common Crawl dataset. This underrepresentation can lead to performance disparities when training general-purpose models.

:p What is the ratio between world population representation and Common Crawl representation for English?
??x
The ratio for English is 0.40, calculated based on its presence (45.88%) compared to its proportion in the world's population (18.15%).

Explanation:
- World population: 8 billion
- Percentage of speakers: 18.15%
- Percentage in Common Crawl: 45.88%

Ratio = \(\frac{\text{Percentage in Common Crawl}}{\text{Percentage of Speakers}} = \frac{45.88\%}{18.15\%} = 2.526 \approx 0.40\) (for comparison)

x??

---

#### Impact on Model Performance

Background context: General-purpose models tend to perform much better for English than under-represented languages due to the dominance of English in internet data.

:p How does the performance of GPT-4 differ between English and an under-represented language like Telugu according to MMLU benchmark?
??x
On the MMLU benchmark, GPT-4 performed significantly better in English than in under-represented languages such as Telugu. This highlights the disparity in model performance across different languages due to their varying degrees of representation in training data.

Explanation:
- The MMLU benchmark includes 14,000 multiple-choice questions covering 57 subjects.
- GPT-4's performance on English questions was notably higher compared to its performance on Telugu questions.

x??

---

#### GPT-4 Performance on MMLU Benchmark
Background context: The passage discusses how GPT-4 performs better on the Multiple Choice Mathematics section of the MMLU benchmark when tested in English compared to other languages. This performance gap is significant enough that even under-representation and structural differences are mentioned as factors.
:p How does GPT-4 perform on the MMLU benchmark for different languages?
??x
GPT-4 performs significantly better in English on the MMLU Multiple Choice Mathematics section compared to other languages like Armenian, Farsi, Burmese, and Amharic. The performance gap is particularly evident when testing on math problems from Project Euler, where GPT-4 solved three times as many problems in English as it did in Armenian or Farsi.
In more detail, GPT-4 failed completely for all six math problems presented in Burmese and Amharic. This disparity suggests that under-representation of certain languages is a significant factor, but the structure and cultural nuances of some languages also contribute to the performance differences.
??x
---

#### Translation Challenges for LLMs
Background context: The text highlights potential issues with translating queries from non-English languages into English before feeding them to the model. These challenges include understanding under-represented languages and losing information during translation.
:p Can we translate all queries from other languages into English, obtain responses, and then translate them back? What are the drawbacks?
??x
While translating queries from other languages into English can seem like a straightforward solution, it has significant drawbacks. The main issues include:
1. **Understanding Under-Represented Languages**: Models need to understand the nuances of under-represented languages well enough to make accurate translations.
2. **Information Loss During Translation**: Translation may lead to loss of context or specific cultural details due to differences in language structures and idioms.

For example, Vietnamese pronouns that denote relationship between speakers are translated into generic "I" and "you," losing the original meaning and nuance.
??x
---

#### Performance Variations Across Languages
Background context: The passage discusses how GPT-4’s performance varies significantly across different languages on the MMLU benchmark. Under-representation of certain languages in training data is mentioned as a factor, but language structure also plays a role.
:p What factors contribute to GPT-4's varying performance across different languages?
??x
Several factors contribute to GPT-4's varying performance across different languages:
1. **Under-Representation**: Certain under-represented languages like Telugu, Marathi, and Punjabi perform worse on the MMLU benchmark.
2. **Language Structure**: The structure of a language can make it harder for the model to learn effectively. For example, Burmese requires significantly more tokens (median length 72) compared to English (median length 7).
3. **Cultural Nuances**: Cultural context and idiomatic expressions can also affect performance.
??x
---

#### Inference Latency and Cost in Non-English Languages
Background context: The text mentions that inference latency and cost for non-English languages are higher due to more tokens required to convey the same meaning. This is illustrated with token length differences between English, Hindi, and Burmese.
:p Why are non-English languages like Burmese more expensive in terms of model inference?
??x
Non-English languages like Burmese are more expensive in terms of model inference because they require significantly more tokens to convey the same meaning. For instance, on the MASSIVE dataset, the median token length is:
- 7 for English
- 32 for Hindi
- 72 for Burmese

This means that GPT-4 takes approximately ten times longer in Burmese than in English for the same content, leading to higher costs per inference.
??x
---

#### Training Focus on Non-English Languages
Background context: The passage highlights active model training efforts focused on non-English languages, particularly Chinese. Other models like ChatGLM and YAYI are mentioned as examples of such focus.
:p What is the current trend in LLM training regarding non-English languages?
??x
The current trend in large language model (LLM) training is a growing focus on non-English languages, especially those that are under-represented. This includes:
1. **Chinese**: Models like ChatGLM and YAYI have been actively trained to support Chinese.
2. **French, Vietnamese, Arabic**: There are also active models for these languages, such as CroissantLLM, PhoGPT, and Jais.

This trend aims to address the performance disparities highlighted in the MMLU benchmark and other tests.
??x
---

#### Domain-Specific Models vs. General-Purpose Models
Background context: This section discusses the differences between general-purpose models like Gemini, GPTs, and Llamas, which can handle a wide range of domains, and domain-specific models that are tailored for particular tasks or fields.

General-purpose models benefit from diverse training data across multiple domains, including coding, law, science, business, sports, and environmental science. Domain-specific models, on the other hand, are crafted to perform well in specific areas like drug discovery, cancer screening, protein structure prediction, etc., which require specialized datasets.

:p How do general-purpose models compare to domain-specific models in terms of their training data?
??x
General-purpose models such as Gemini, GPTs, and Llamas are trained on a wide variety of domains including coding, law, science, business, sports, and environmental science. This broad training allows them to handle questions across multiple fields effectively.

Domain-specific models, like AlphaFold for protein structure prediction or Med-PaLM2 for medical queries, are specifically tailored to perform well in narrow but deep areas that general-purpose models might struggle with due to lack of specialized training data.
x??

---

#### Common Crawl and Domain Distribution
Background context: The text mentions an analysis by the Washington Post on the distribution of domains present in the C4 dataset from Common Crawl. This analysis helps understand how diverse the training data is for certain models.

:p What does the distribution of domains in Common Crawl reveal about model training?
??x
The distribution of domains in Common Crawl reveals which topics and fields are well-represented in the training data used by models like Gemini, GPTs, and Llamas. For instance, if a domain is heavily present (like business or science), the model might perform better on questions related to that field due to extensive exposure during training.

This can be seen visually through Figure 2-3 which shows the distribution of domains in the C4 dataset.
x??

---

#### Benchmarks for Domain-Specific Performance
Background context: The text provides an example of how models like CLIP and OpenCLIP perform on specific image datasets, indicating their strengths and limitations.

:p How can benchmarks help determine a model's performance on domain-specific tasks?
??x
Benchmarks provide a way to measure a model’s proficiency in specific areas by testing it against well-defined criteria. For instance, Table 2-3 shows how CLIP and OpenCLIP perform on datasets related to birds, flowers, cars, etc., which can indicate the models' strengths and limitations in recognizing these categories.

For domain-specific tasks like drug discovery or cancer screening, such benchmarks can help identify whether a model is suitable by showing its accuracy in relevant categories.
x??

---

#### Training Data Impact
Background context: This section discusses how the training data influences a model's performance on various domains. The text mentions that general-purpose models are trained on diverse datasets, while domain-specific models are curated for particular fields.

:p How does the quality and type of training data affect a model’s performance?
??x
The quality and type of training data significantly impact a model's performance across different domains. General-purpose models benefit from broad exposure to various subjects through large, diverse datasets like Common Crawl. However, domain-specific tasks require specialized knowledge that may not be present in general internet data.

For example, drug discovery involves protein, DNA, and RNA data, which are unlikely to appear frequently enough in public web content for a model to learn effectively from it.
x??

---

