# Flashcards: 2A014 (Part 25)

**Starting Chapter:** 106-Keep It Simple

---

#### Experimentation Tips Overview
In this section, the authors discuss guidelines for experimentation and rapid iteration during the development of ranking models. The primary focus is on making experimental code efficient without compromising too much on quality to facilitate quick testing and hypothesis validation.

:p What are the main tips provided by the authors for conducting experiments in ranking models?
??x
The main tips include:
1. Recognizing that experimental code is different from production code; it should prioritize speed of experimentation over robustness.
2. Deciding whether a piece of code needs thorough testing based on its role in hypothesis testing versus long-term use.

Example scenarios where these tips are applicable might involve quickly prototyping loss functions without full regression testing, or writing metrics for performance evaluation that are discarded after the experiment concludes.

```java
public class ExperimentCode {
    // Code here is meant to test a hypothesis and may not be fully robust.
    public void quickTestLossFunction() {
        // Prototype implementation of loss function for rapid experimentation
    }
}
```
x??

---
#### Rapid Iteration Strategy
The strategy focuses on achieving maximum velocity in the development process by balancing speed with maintainability. This involves making decisions about code quality and testing based on the immediate needs of hypothesis validation.

:p What does the term "maximum velocity" refer to in the context of experimental coding?
??x
Maximum velocity refers to the ability to rapidly develop, test, and iterate on ideas without being overly constrained by traditional engineering standards for robustness and maintainability. This approach is suitable for initial exploratory stages where quick results are more important than maintaining perfect code quality.

```java
public class VelocityExample {
    public void exploreIdeas() {
        // Code here focuses on exploring new ideas quickly, possibly with less testing.
    }
}
```
x??

---
#### General Guidelines for Experimentation
The guidelines suggest that experimental code should be fast and flexible to enable quick validation of hypotheses. The authors emphasize the importance of context in deciding whether rigorous testing is necessary or if a faster but simpler approach suffices.

:p How does the concept of "code quality" differ in experimental versus production settings?
??x
In experimental settings, the emphasis is on velocity and flexibility rather than long-term maintainability. Code may be simpler, quicker to write, and less rigorously tested because its primary purpose is to test hypotheses rapidly. Production code, on the other hand, requires higher standards for robustness, reliability, and maintainability.

```java
public class ExperimentCodeQuality {
    public void prototypeLossFunction() {
        // Simple implementation without extensive validation.
    }

    public void productionLossFunction() {
        // Robust implementation with comprehensive tests.
    }
}
```
x??

---

#### Keep It Simple
In terms of the overall structure of research code, simplicity should be prioritized over complexity during the early stages of a project. Code that is easily readable and simple for debugging is more valuable than highly reusable but complex code at this stage. Refactoring too early can slow down the development velocity.
:p Why should you keep your initial research code simple?
??x
At the beginning of a project, the structure of the model, data ingestion, and interactions between various parts of the system are still being worked out. Many changes will occur in the early stages, making it difficult to implement robust reusability at this point. Keeping the code simple facilitates easier debugging and modification as you work through uncertainties.
x??

---

#### Debug Print Statements
Debug print statements are crucial for inspecting messy real-world data and understanding how transformations and models behave during experimentation. These help in identifying issues early on, ensuring that your input and output schemas match expectations across different components of the system.

:p How do debug print statements assist in managing messy data?
??x
By printing out samples of the data, you can visually inspect them for missing fields or unexpected values. This helps in crafting appropriate data pipelines and transformations before feeding the model. Additionally, debugging print statements help verify that the output of models is as expected.
x??

---

#### Defer Optimization
Optimization should be deferred until after the bulk of experimentation has been completed. Early optimization can distract from the primary goal of rapid experimentation and might not yield long-term benefits if the code or architecture changes significantly.

:p Why should you defer optimization in research code?
??x
Optimization early on may not make sense because other parts of the system could be slower bottlenecks, making further optimizations elsewhere more critical. Additionally, optimizing a part that ends up being refactored away can waste effort. Lastly, optimized code might constrain future modifications and design choices.
x??

---

#### JAX NaN Debugging
In JAX, enabling debug print statements during JIT compilation helps detect NaN errors by rerunning the function and printing tensor values where necessary.

:p How can you enable NaN debugging in JAX?
??x
You can enable NaN debugging in JAX using the following lines of code:

```python
from jax import config
config.update("jax_debug_nans" , True)

@jax.jit
def f(x):
  jax.debug.print("Debugging {x}", x=x)
```

This configuration will rerun a jitted function if it finds any NaNs, and the debug print function will print the value of tensors even inside JIT compilation. Regular print statements do not work within JIT as they are non-compilable commands skipped during tracing.
x??

---

#### Experimentation Tips
Experimentation in research code should focus on rapid prototyping and testing rather than initial optimization. The primary goal is to understand the system behavior, validate models, and iterate quickly.

:p What is an important tip for experimentation in research code?
??x
Do not optimize too early unless it hinders research velocity. Early optimization might be premature or distract from more critical issues that need addressing first. Focus on making your code readable and easy to debug during the initial phases of development.
x??

---

#### Keeping Track of Changes
In research, managing numerous variables can be challenging. With large datasets and many runs required to identify the best changes, it is essential to systematically alter parameters while tracking their effects.
:p How do you manage a multitude of variables during research code modifications?
??x
To manage multiple variables effectively, use tools like Weights & Biases that allow you to track both code changes and parameter settings. This helps in reproducing experiments and analyzing results.
```python
import wandb

# Initialize W&B project
wandb.init(project="my_project")

# Log parameters and metrics
wandb.log({"learning_rate": 0.01, "batch_size": 32})
```
x??

---

#### Feature Engineering in Applied Research
Feature engineering is crucial in applied research where practical outcomes are prioritized over theoretical perfection. Handcrafted features can be added to enhance model performance, especially when data is limited or time constraints exist.
:p How does feature engineering contribute to model performance?
??x
Feature engineering helps by adding domain-specific knowledge into the model training process. For example, in recommender systems, you could create a boolean feature indicating if an item's attribute (like artist or album) matches something in the user’s profile. This can speed up convergence and complement latent feature learning.
```python
def engineer_features(item_data, user_profile):
    features = []
    for item in item_data:
        match_artist = int(user_profile['artist'] == item['artist'])
        match_album = int(user_profile['album'] == item['album'])
        features.append({'match_artist': match_artist, 'match_album': match_album})
    return features
```
x??

---

#### Ablating Hand-Engineered Features
Regularly evaluating and pruning hand-engineered features ensures they remain relevant. By periodically removing some features from the model training process, you can determine if these features still provide value or have become obsolete.
:p How do you perform feature ablation?
??x
Feature ablation involves holding back certain engineered features during a model's training phase to check their impact on performance metrics. This helps in identifying whether these features are still beneficial or need to be discarded.
```python
def ablate_features(features, ablate_list):
    filtered_features = [f for f in features if f['feature_name'] not in ablate_list]
    return filtered_features

# Example usage:
ablated_features = ablate_features(item_data, ['match_artist', 'match_album'])
```
x??

---

