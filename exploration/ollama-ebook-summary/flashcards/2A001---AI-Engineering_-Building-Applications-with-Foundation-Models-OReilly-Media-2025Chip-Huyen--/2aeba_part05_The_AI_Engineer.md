# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 5)

**Starting Chapter:** The AI Engineering Stack

---

#### Data Flywheel Concept
AI startups often refer to a "data flywheel," which is about gathering user data, improving product performance, and attracting more users. This cycle can create a competitive advantage for a startup.

:p What does the phrase "data flywheel" mean in the context of AI startups?
??x
The term "data flywheel" refers to a process where a company collects data from its users, uses that data to improve its product or service, and as a result attracts more users. This cycle can create a competitive advantage by continuously improving the quality of the product based on user interaction.

Example: A startup might use user feedback to improve a chatbot's responses, leading to happier customers who return for more interactions, thus generating even more data.
x??

---

#### Competitive Advantages in AI
In AI, there are three main types of competitive advantages: technology, data, and distribution. Foundation models often have similar core technologies, giving big companies an edge in distribution. However, a startup can differentiate itself by gathering sufficient usage data to continually improve its product.

:p What are the three general types of competitive advantages in AI?
??x
The three general types of competitive advantages in AI are:
1. Technology: This involves the core algorithms and models used.
2. Data: This refers to the amount and quality of data available for training and improving products.
3. Distribution: This is about reaching a wide audience, often through existing platforms or direct marketing efforts.

Example: Big companies like Google might have more existing data due to their scale but if a startup can gather sufficient usage data first, it can build a competitive advantage based on continuous improvement.
x??

---

#### Calendly and Mailchimp Analogy
Calendly could have been integrated into Google Calendar as a feature. Similarly, Photoroom might have been included in Google Photos. These examples illustrate that some successful startups started with features that larger companies overlooked.

:p Provide an example of how a smaller startup could potentially overtake a bigger competitor.
??x
An example is Calendly. It was a standalone product but could have been integrated into Google Calendar as a feature. However, it became a successful standalone product because it provided additional value beyond what Google Calendar offered. Similarly, Photoroom could have been part of Google Photos, but instead, it built its own platform to offer specific functionalities that were not present in Google Photos.

Example: If Calendly had become a part of Google Calendar with limited features, it might not have achieved the same level of success as when it remained a standalone product.
x??

---

#### Setting Expectations for AI Applications
To measure success in building an AI application, startups need to define clear business metrics. For instance, if the application is a customer support chatbot, key performance indicators (KPIs) could include automation percentages, message processing efficiency, response speed, and human labor savings.

:p How can you measure the success of an AI chatbot?
??x
To measure the success of an AI chatbot, you need to define clear business metrics. Key KPIs might include:
- What percentage of customer messages do you want the chatbot to automate?
- How many more messages should the chatbot allow you to process?
- How much quicker can you respond using the chatbot?
- How much human labor can the chatbot save you?

Example: If a chatbot automatically handles 60% of customer queries, processes an additional 5,000 messages per day, reduces response time by 8 hours daily, and saves 10 hours of human labor per week, these are key metrics to track.
x??

---

#### Useful Thresholds for AI Products
When building an AI application, it's crucial to define clear expectations on the product’s usefulness threshold. This helps ensure that the product is not put in front of customers before it meets a certain standard.

:p What is a useful threshold in the context of AI products?
??x
A useful threshold is the level of quality or performance a product must achieve before being considered ready for users. It ensures that the product provides enough value to warrant its use.

Example: For a chatbot, the usefulness threshold could include metrics like:
- Quality metrics: Accuracy and relevance of responses.
- Latency metrics: Time to first token (TTFT), time per output token (TPOT), total latency.

If these thresholds are not met, the product should not be released. For instance, if your customer requests have a median response time of an hour with humans handling them, any chatbot that responds faster than this could meet the threshold.
x??

---

#### Cost Metrics for Inference Requests
Cost metrics are crucial when evaluating the efficiency and feasibility of AI applications. They measure how much it costs per inference request, which is essential for understanding operational expenses. Other relevant metrics include interpretability (the ease with which humans can understand why a model made a specific decision) and fairness (ensuring that the model does not discriminate against certain groups).

:p What are cost metrics in the context of AI applications?
??x
Cost metrics refer to the financial cost associated with each inference request made by an AI application. This includes both the direct costs like cloud compute services and any indirect costs such as data storage and network usage.
x??

---

#### Evaluating Existing Models for Goals
The initial step in planning an AI product is understanding the capabilities of existing models. By evaluating these models, you can determine how much work needs to be done to meet your goals. For instance, if a model can automate 30% of customer support tickets and your goal is to automate 60%, then you might need less effort compared to starting from scratch.

:p How does evaluating existing models help in setting AI product goals?
??x
Evaluating existing models helps set realistic AI product goals by providing insight into the current state-of-the-art capabilities. It allows you to gauge how much additional development is required to meet your objectives. For example, if a model can automate 30% of customer support tickets and your goal is to automate 60%, this evaluation might show that only 30% more effort is needed rather than starting from zero.
x??

---

#### Milestone Planning for AI Products
Once you have set measurable goals, creating a plan to achieve them becomes essential. The effectiveness of the plan depends on where you start. If your off-the-shelf model can already partially meet some of your goals, then less effort might be required compared to starting from scratch.

:p What is the importance of planning for AI products?
??x
Milestone planning is crucial because it breaks down the complex task of developing an AI product into manageable steps. This helps in understanding the resources and efforts needed at each stage. Starting with existing models can significantly reduce the effort, making the project more feasible. For example, if a model can automate 30% of customer support tickets and your goal is to automate 60%, you only need to add functionality for automating an additional 30%, rather than starting from zero.
x??

---

#### Last Mile Challenges in AI Product Development
Initial success with foundation models might be misleading, as the ease of building a demo does not necessarily predict the effort required to build a full product. The "last mile" challenge refers to the difficulty in scaling and improving an initial working model to meet all user needs.

:p What are last mile challenges in AI product development?
??x
Last mile challenges refer to the difficulties encountered when moving from a basic, functional prototype to a fully developed, robust product that meets all user requirements. These challenges often involve refining the model's performance, addressing edge cases, and ensuring reliability. For instance, transitioning from an initial 80% functionality level to surpassing 95% can be much harder than expected.
x??

---

#### Maintenance of AI Products
Maintenance is a critical aspect of AI product development that extends beyond achieving initial goals. You need to consider how the product will evolve and how it should be maintained over time. The fast-paced nature of AI means that maintaining an AI product requires constant updates and adaptations.

:p Why is maintenance important for AI products?
??x
Maintenance is crucial because AI models continue to evolve, with new advancements in technology and techniques becoming available regularly. Maintaining an AI product involves keeping it up-to-date with these advancements, addressing any bugs or issues, and ensuring continued performance and reliability. This ongoing process helps the product stay relevant and effective over time.
x??

---

#### Example of Initial Success Misleading
Initial success with foundation models can be misleading because achieving a certain level of functionality is easier than improving that initial model to meet all user needs. The example provided in UltraChat highlights this challenge, showing how quickly developers underestimated the time needed for further improvements.

:p How does initial success with foundation models often mislead development efforts?
??x
Initial success with foundation models can create an illusion of easy progress and underestimate the time and effort required to fully develop a product. For instance, UltraChat took only one month to achieve 80% of their desired experience but then found it took four more months to surpass 95%. This demonstrates that initial successes may not accurately predict the challenges and time needed for subsequent improvements.
x??

---

#### Evolution of Inference Costs
Inference costs are decreasing rapidly as AI models improve. The cost of inference has dropped significantly over a short period, making it cheaper and faster to compute model outputs.

:p How have inference costs evolved in recent years?
??x
Inference costs have dramatically decreased due to advancements in AI technology. Between 2022 and 2024, the cost per unit of model performance on MMLU has dropped significantly, making computations more affordable and efficient. For example, Figure 1-11 illustrates this trend, showing that inference costs rapidly decline over time.
x??

---

#### Workflow Friction in AI Applications

Background context: The text discusses how changes in technology and regulations can create challenges for teams working on AI applications. These changes might initially seem beneficial but could later become drawbacks, requiring frequent reassessment of costs and benefits.

:p What are some examples of workflow friction mentioned in the text?
??x
The text mentions several instances where initial decisions may turn out to be suboptimal over time:
- Choosing an in-house model for cost reasons that becomes more expensive due to changes in provider pricing.
- Investing in a third-party solution only to face business failures from providers.
- Regulatory changes, such as GDPR, which can significantly impact costs and compliance.
- Sudden bans on GPU vendors leading to operational disruptions.

x??

---

#### Cost-Benefit Analysis of Technology Investments

Background context: The text emphasizes the need for ongoing cost-benefit analysis due to rapidly changing technology landscapes. Initial decisions might seem optimal but can become suboptimal over time as conditions change.

:p Why is a constant cost-benefit analysis necessary in AI application development?
??x
A constant cost-benefit analysis is crucial because:
- Technology prices and availability can fluctuate, making initial choices potentially outdated.
- Market conditions, including competition and regulations, evolve continuously.
- Initial decisions might become less favorable as technology improves or external factors change.

Example: Initially deciding to build a model in-house based on current costs, but later finding that providers have reduced their pricing by half.

x??

---

#### API Convergence and Model Interoperability

Background context: The text notes that as model providers standardize on APIs, it becomes easier to switch between different models. However, developers still need to adjust workflows according to the quirks of new models.

:p How does API convergence make switching between AI models easier?
??x
API convergence simplifies switching between AI models by providing a standardized interface. This allows developers to easily replace one model with another without changing their existing codebase or infrastructure significantly.
For example, if you have an application using Model A and want to switch to Model B, both models might expose similar APIs, making the transition seamless.

x??

---

#### Infrastructure Challenges in AI Engineering

Background context: The text highlights the importance of proper versioning and evaluation infrastructures when dealing with evolving AI technologies. Without such infrastructure, changes can cause significant headaches for developers.

:p What are some key challenges related to infrastructure in AI engineering?
??x
Key challenges include:
- Managing different versions of models.
- Evaluating model performance accurately.
- Adapting workflows and prompts according to new models' quirks.
Without proper infrastructure, these challenges can lead to inefficiencies and increased development time.

Example: A need for version control systems and automated testing frameworks to manage multiple model versions effectively.

x??

---

#### AI Engineering vs. ML Engineering

Background context: The text differentiates between AI engineering and traditional ML engineering, noting that they share significant overlap but have distinct roles in the AI application building process.

:p How does AI engineering differ from traditional ML engineering?
??x
AI engineering differs from traditional ML engineering primarily by focusing on:
- Dealing with more complex and dynamic models.
- Managing regulatory compliance issues related to national security concerns.
- Ensuring data privacy and intellectual property rights.
While both roles involve similar technical skills, AI engineers often work in environments with stricter legal and ethical considerations.

x??

---

#### Engineering Stack for Building AI Applications

Background context: The text introduces the concept of an AI engineering stack, emphasizing that while there is a lot of hype around new tools and techniques, understanding the fundamental building blocks is essential.

:p What are the key components of the AI engineering stack?
??x
The key components include:
- Data management.
- Model training and evaluation.
- Deployment and monitoring.
- Version control and infrastructure support.
These components form the backbone of any AI application, ensuring it can scale and adapt to changing conditions.

Example: A typical stack might involve using tools like TensorFlow for model training, Docker for deployment, and Git for version control.

x??

---

#### AI Application Stack Layers
Background context: The text describes a three-layer stack for AI applications, which are Application Development, Model Development, and Infrastructure. Each layer has specific responsibilities and roles involved.

:p What are the three layers of the AI application stack?
??x
The three layers of the AI application stack are:
1. **Application Development**: This involves using pre-existing models to develop applications by providing good prompts and necessary context.
2. **Model Development**: This includes tools for developing new models, such as frameworks for modeling, training, finetuning, and inference optimization. It also encompasses dataset engineering.
3. **Infrastructure**: At the bottom of the stack, this layer focuses on tooling for model serving, data and compute management, and monitoring.

x??

---
#### Application Development Layer
Background context: The application development layer is where anyone can use readily available models to develop applications. It's a rapidly evolving field that requires good prompts, necessary context, and rigorous evaluation of the applications developed.

:p What does application development in AI involve?
??x
Application development in AI involves providing a model with well-crafted prompts and necessary context. The process also requires thorough evaluation to ensure that the applications are effective and user-friendly.
```java
public class ApplicationDeveloper {
    public void developApplication(String prompt, Context context) {
        // Use pre-existing models to develop an application based on the prompt and context provided.
        Model model = fetchModel(prompt);
        Application app = generateApplication(model, context);
        evaluateApplication(app); // Ensure the application meets quality standards
    }
}
```
x??

---
#### Model Development Layer
Background context: The model development layer provides tools for developing new models, including frameworks for modeling, training, finetuning, and inference optimization. It also involves dataset engineering.

:p What does the model development layer include?
??x
The model development layer includes:
- Frameworks for modeling, training, finetuning, and inference optimization.
- Dataset engineering to centralize data used in model development.

x??

---
#### Infrastructure Layer
Background context: The infrastructure layer is crucial for deploying models at scale. It involves managing resources such as data and compute, and monitoring the performance of deployed models.

:p What does the infrastructure layer include?
??x
The infrastructure layer includes:
- Tooling for model serving to deploy models in production environments.
- Data management tools to handle large datasets.
- Compute management to allocate appropriate resources based on needs.
- Monitoring systems to track the performance and health of deployed models.

x??

---
#### GitHub Repository Analysis
Background context: In March 2024, a search was conducted on GitHub for AI-related repositories with at least 500 stars. The analysis included applications and model development layers. A total of 920 repositories were found across different categories.

:p What data was analyzed to understand the ecosystem of foundation models?
??x
The data analyzed to understand the ecosystem of foundation models includes:
- GitHub repositories with at least 500 stars.
- Categories such as applications, models, and infrastructure tools.
- A total of 920 repositories were found across these categories.

x??

---
#### Repositories Over Time
Background context: The analysis showed a cumulative count of repositories in each category month-over-month. This data gives insight into the growth and evolution of AI-related projects on GitHub.

:p How was the trend of repository growth tracked over time?
??x
The trend of repository growth was tracked by analyzing the cumulative count of repositories across different categories (applications, models, infrastructure) month-over-month. This provided insights into how the ecosystem has evolved with foundation models.

x??

---

#### AI Tooling Growth Post 2023 Introduction

Background context: The data shows a significant increase in the number of AI toolings introduced after the introduction of Stable Diffusion and ChatGPT. The highest increases were observed in applications and application development, while the infrastructure layer saw less growth.

:p What was the notable trend observed in AI tooling usage post 2023?

??x
The notable trend observed is a significant increase in the number of AI tools, with the highest growth seen in application-related areas. Infrastructure improvements were also noted but to a lesser extent.
x??

---

#### Growth Discrepancy Between Layers

Background context: Despite changes in models and applications, the core infrastructural needs—such as resource management, serving, monitoring—remain the same.

:p Why did the infrastructure layer see less growth compared to other layers?

??x
The infrastructure layer saw less growth because while there were advancements in models and applications, the fundamental requirements for managing resources, serving, and monitoring remained unchanged.
x??

---

#### ML vs. AI Engineering Principles

Background context: The principles of building AI applications are similar to those of traditional machine learning (ML) engineering, but with key differences.

:p How do enterprise use cases for AI applications differ from classical ML engineering?

??x
Enterprise use cases for AI applications still require solving business problems by mapping between business metrics and ML metrics. Systematic experimentation is essential, whether it involves hyperparameters in classical ML or models, prompts, retrieval algorithms, and more in foundation models.
x??

---

#### Key Differences Between AI and ML Engineering

Background context: Building applications using foundation models today differs from traditional ML engineering in three major ways.

:p What are the three major differences between AI engineering and traditional ML engineering?

??x
The three major differences are:
1. Traditional ML requires training your own models, whereas AI engineering uses pre-trained models.
2. Foundation models consume more compute resources and incur higher latency, requiring efficient training and inference optimization.
3. Open-ended outputs in foundation models make evaluation a more challenging task.
x??

---

#### GPU Requirements for Large-Scale Computing

Background context: The increased demand for larger compute clusters necessitates expertise in working with GPUs.

:p Why is there an increased need for engineers who can work with large GPU clusters?

??x
There's an increased need because many companies now require managing more GPUs and bigger compute clusters than before. This highlights the importance of having engineers skilled in handling such resources.
x??

---

#### Evaluation Challenges in AI Engineering

Background context: The evaluation process is significantly more complex due to open-ended outputs from foundation models.

:p What makes model evaluation a larger challenge in AI engineering?

??x
Evaluation becomes much harder because foundation models can produce open-ended outputs, which offer flexibility but also complicate the assessment of their performance and utility.
x??

---

#### Example of GPU Cluster Management

Background context: An example illustrates the skill gap when scaling from small to large GPU clusters.

:p What does the example about a Fortune 500 company's team signify regarding GPU cluster management?

??x
The example highlights that while a team may be proficient with handling 10 GPUs, they may lack experience with managing much larger GPU clusters (e.g., 1,000 GPUs). This underscores the need for specialized expertise in large-scale GPU cluster management.
x??

---

#### Model Adaptation Techniques
Background context explaining model adaptation techniques, including prompt-based methods and finetuning. These techniques are used to adapt models for specific tasks without necessarily updating the underlying weights of the model or by making changes directly to the model itself.

:p What are the two main categories of model adaptation techniques mentioned in the text?
??x
The two main categories of model adaptation techniques are prompt-based methods and finetuning. Prompt-based methods, such as prompt engineering, adapt a model without updating the model weights. They involve giving instructions and context to the model instead of changing it directly. Finetuning requires updating model weights by making changes to the model itself. This method is more complex and data-intensive but can significantly improve the quality, latency, and cost of models.
x??

---

#### Prompt-Based Techniques
Background context discussing prompt-based techniques as a way to adapt models without updating their weights. These methods are easier to start with and require less data.

:p What does prompt engineering involve in adapting a model?
??x
Prompt engineering involves adapting a model by providing instructions and context rather than changing the model's structure or weights. This approach is simpler to implement, requires fewer resources, and allows for quick experimentation across different models. It enables you to experiment with more models and increases the chances of finding one that performs exceptionally well for your application.
x??

---

#### Finetuning
Background context explaining finetuning as a method to update model weights by making changes directly to the model. This technique is more complex and requires more data but can significantly improve model performance.

:p What does finetuning involve in adapting a model?
??x
Finetuning involves updating the model weights by making direct changes to the model itself. Unlike prompt-based methods, this approach modifies the internal parameters of the model. While it is more complex and requires more data, finetuning can significantly enhance the quality, latency, and cost-effectiveness of models, especially for tasks that require high performance or are not adequately addressed by simpler methods.
x??

---

#### Model Development Layer
Background context describing the responsibilities in developing an AI application, including modeling and training, dataset engineering, inference optimization, and evaluation.

:p What are the three main responsibilities of model development?
??x
The three main responsibilities of model development include:
1. **Modeling and Training**: This involves creating a model architecture, training it using appropriate datasets, and finetuning the model to improve performance.
2. **Dataset Engineering**: This step focuses on preparing and processing data for use in the models, including cleaning, normalization, augmentation, and labeling.
3. **Inference Optimization**: This responsibility involves optimizing the model's deployment to ensure efficient and effective predictions during real-time or production use.

These responsibilities collectively ensure that an AI application can be developed and deployed effectively.
x??

---

#### Modeling and Training
Background context on modeling and training, including examples of tools like TensorFlow, Hugging Face’s Transformers, and Meta’s PyTorch. It also mentions the need for specialized ML knowledge to develop models.

:p What are some popular tools used in modeling and training?
??x
Some popular tools used in modeling and training include:
- **Google's TensorFlow**: A powerful platform for building and deploying machine learning models.
- **Hugging Face’s Transformers**: A library that provides state-of-the-art models and pipelines for natural language processing tasks.
- **Meta’s PyTorch**: An open-source machine learning library based on the Torch library, widely used in deep learning research.

These tools provide a range of functionalities from model building to training and inference optimization. Developers need specialized ML knowledge to effectively use these tools, including understanding different types of algorithms like clustering, logistic regression, decision trees, collaborative filtering, as well as neural network architectures such as feedforward, recurrent, convolutional, and transformer models.
x??

---

#### Evaluation
Background context on evaluation in the application development layer, noting that it is discussed further in a separate section.

:p Why is evaluation important in model development?
??x
Evaluation is crucial in model development because it helps assess the performance of the model against predefined metrics or criteria. While most people will encounter evaluation first in the application development layer, understanding its role in model development ensures comprehensive testing and validation before deployment. Evaluation helps identify any shortcomings in the model's performance, ensuring that it meets the required standards for accuracy, reliability, and efficiency.
x??

---

---
#### Pre-training
Pre-training refers to training a model from scratch—the model weights are randomly initialized. For large language models (LLMs), pre-training often involves training a model for text completion. Out of all training steps, pre-training is often the most resource-intensive by a long shot. For the InstructGPT model, pre-training takes up to 98 percent of the overall compute and data resources.
:p What does pre-training involve?
??x
Pre-training involves training a model from scratch with randomly initialized weights. This phase focuses on learning general knowledge or patterns that can be applied across various tasks. It is highly resource-intensive due to the extensive computation required for large models like LLMs.
x??

---
#### Finetuning
Finetuning means continuing to train a previously trained model—the model weights are obtained from the previous training process. Because the model already has certain knowledge from pre-training, finetuning typically requires fewer resources (e.g., data and compute) than pre-training.
:p What is finetuning?
??x
Finetuning involves further training an existing model that has been pretrained to refine its performance on specific tasks. This phase leverages the pre-trained knowledge to adapt the model more effectively for a particular application with less computational resources compared to the initial pre-training process.
x??

---
#### Post-training
Post-training and finetuning are often used interchangeably, but sometimes people might use them differently to signify different goals. Conceptually, post-training is done by model developers, while finetuning can be performed by application developers. It’s usually post-training when a model like InstructGPT is improved for following instructions before release.
:p What does the term "post-training" typically refer to?
??x
Post-training typically refers to further optimizing or improving an already pretrained model for specific tasks, often done by model developers. This could include adjustments that make the model better at certain aspects of its function, such as enhancing instruction-following capabilities in InstructGPT.
x??

---
#### Training vs. Prompt Engineering
Training is a broader term that can encompass pre-training, finetuning, and post-training phases. However, some people use training to refer to prompt engineering, which isn't technically correct. Prompt engineering involves teaching a model via the context input into the model.
:p What's the difference between "training" and "prompt engineering"?
??x
Training refers to the process of adjusting model weights through various stages like pre-training, finetuning, or post-training. On the other hand, prompt engineering involves shaping the model's behavior by providing specific instructions or contexts rather than directly altering its weights.
x??

---

---
#### Dataset Engineering
Background context explaining dataset engineering, including its importance and differences from traditional ML engineering. Foundation models require more open-ended data annotation compared to close-ended models like spam classification.

:p What is dataset engineering?
??x
Dataset engineering involves curating, generating, and annotating the data needed for training and adapting AI models. It's particularly challenging for foundation models due to their open-ended nature, where annotating queries requires significant effort compared to predefined values.

For example, while it might be straightforward to classify an email as "spam" or "not spam," writing a coherent essay is much more complex and time-consuming.

```java
// Pseudocode for basic dataset annotation process
public class AnnotationTool {
    public void annotateData(String data) {
        // Process the input data (e.g., text, images)
        if (data.contains("harmful content")) {
            label = "toxic";
        } else if (data.matches(patternForSpam)) {
            label = "spam";
        } else {
            label = "neutral";
        }
        // Output the labeled data for further processing
    }
}
```
x??

---
#### Modeling and Training with Foundation Models vs. Traditional ML

Background context explaining how modeling and training have changed from traditional ML to foundation models, highlighting that ML knowledge is not as critical for foundation models but still important.

:p How does modeling and training differ between foundation models and traditional ML?

??x
In traditional machine learning (ML), building a model from scratch requires significant expertise in ML techniques. However, with foundation models, the need for deep ML knowledge diminishes because these models are often fine-tuned or adapted using smaller datasets through simpler methods.

For example:
- **Traditional ML**: Training a model from scratch might require understanding complex algorithms and large amounts of data.
- **Foundation Models**: Fine-tuning or adapting an existing large pre-trained model might involve less intricate ML knowledge, although it still requires some level of expertise in the domain and understanding of how to use these models effectively.

```java
// Pseudocode for traditional ML vs. Foundation Model training
public class TrainingExample {
    public void trainFromScratch(Dataset dataset) {
        // Complex algorithmic training process required here
    }

    public void fineTuneModel(Dataset dataset, FoundationModel model) {
        // Smaller dataset and simpler methods used here
    }
}
```
x??

---
#### Inference Optimization

Background context explaining the importance of inference optimization for both traditional ML and foundation models, with a particular emphasis on the challenges faced by foundation models due to their autoregressive nature.

:p What is inference optimization?

??x
Inference optimization involves making AI models faster and cheaper. It has always been important in machine learning (ML) engineering because users want fast models, and companies benefit from cost savings. With the rise of foundation models, which can be computationally intensive and time-consuming to generate outputs, inference optimization becomes even more critical.

For instance, autoregressive generation processes used by many foundation models (like text generation) require multiple sequential steps, each taking a certain amount of time. Reducing latency is crucial for maintaining user satisfaction in real-time applications.

```java
// Pseudocode for inference optimization techniques
public class InferenceOptimizer {
    public String optimizeInference(String input, FoundationModel model) {
        // Techniques like beam search or other optimization algorithms can be used here
        return optimizedOutput;
    }
}
```
x??

---
#### Data Needs for Different Adapter Techniques

Background context explaining how the amount of data required varies depending on whether you are training a model from scratch, fine-tuning an existing model, or using prompt engineering.

:p How does the amount of data needed differ between different adapter techniques?

??x
The amount of data required significantly depends on the adapter technique used:

- **Training a Model from Scratch**: Requires large amounts of high-quality labeled data.
- **Fine-Tuning**: Requires less data but still needs to be carefully curated and relevant.
- **Prompt Engineering**: Typically requires minimal data, focusing more on well-crafted prompts.

For example:
```java
// Pseudocode for different data requirements
public class DataRequirement {
    public int getRequiredData(int adapterTechnique) {
        if (adapterTechnique == 0) { // Training from scratch
            return 100000; // Large dataset required
        } else if (adapterTechnique == 1) { // Fine-tuning
            return 50000; // Smaller but still significant dataset
        } else { // Prompt Engineering
            return 1000; // Minimal data needed
        }
    }
}
```
x??

---

#### Evaluation of Foundation Models
Evaluation is a critical process to mitigate risks and uncover opportunities, essential throughout the model adaptation process. It's necessary for selecting models, benchmarking progress, determining deployment readiness, and detecting issues or improvement opportunities.

Evaluation becomes even more important with foundation models due to their open-ended nature and expanded capabilities. Traditional ML tasks often have expected ground truths that can be compared against model outputs, but this is not always the case with open-ended tasks like chatbots, where many possible responses make it challenging to curate exhaustive lists of ground truths.

:p What are some challenges in evaluating foundation models?
??x
The challenges include the open-ended nature of tasks and the expanded capabilities of the models. For example, in tasks like chatbots, there are numerous potential responses, making it difficult to create comprehensive ground truth data. Additionally, different adaptation techniques can affect model performance differently.
x??

---
#### Prompt Engineering for Foundation Models
Prompt engineering involves using carefully crafted prompts to elicit desired behaviors from AI models without changing the underlying model weights. This technique is crucial because different prompts can significantly impact model performance.

Google's Gemini evaluation demonstrated the effectiveness of prompt engineering, where using a CoT@32 (Chain-of-Thought @ 32) technique improved Gemini Ultra’s MMLU (Multimodal Math and Language Understanding) performance from 83.7% to 90.04%.

:p How did Google's Gemini evaluation highlight the impact of prompt engineering?
??x
Google used a CoT@32 prompt engineering technique, which showed Gemini more examples than ChatGPT, leading to better performance on the MMLU benchmark. Specifically, when both models were shown five examples, ChatGPT performed better, but with 32 examples, Gemini Ultra's performance improved significantly.
x??

---
#### Context Construction in Prompt Engineering
Context construction is a part of prompt engineering that involves providing AI models with necessary context and tools to perform complex tasks effectively, especially those involving long contexts. This might require additional systems like memory management to help the model track its history.

:p What does context construction involve in prompt engineering?
??x
Context construction includes giving the AI model the necessary background information, tools, and possibly a memory management system to handle long-term dependencies or histories, ensuring it can perform tasks accurately.
x??

---
#### AI Interface Development for Applications
AI interfaces enable end users to interact with AI applications. With foundation models, anyone can build such applications, whether as standalone products or integrated into other platforms.

:p What is an AI interface in the context of building AI applications?
??x
An AI interface is a means for end users to interact with AI applications. It allows users to input queries or commands and receive responses from the AI model. With foundation models, this interaction can be implemented as standalone products or embedded into existing platforms.
x??

---
#### Differentiation through Application Development
In traditional ML engineering, model quality differentiates teams building proprietary models. However, with foundation models where many use the same underlying model, differentiation comes from application development layers such as evaluation, prompt engineering, and AI interface.

:p How does the approach to differentiation change when using foundation models?
??x
With foundation models, the focus shifts to application development layers like evaluation, prompt engineering, and AI interface rather than just improving the quality of proprietary models. These layers allow teams to differentiate their applications through better user experience, more effective task completion, and tailored interactions.
x??

---

