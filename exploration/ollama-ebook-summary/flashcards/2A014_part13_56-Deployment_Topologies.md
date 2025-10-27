# Flashcards: 2A014 (Part 13)

**Starting Chapter:** 56-Deployment Topologies

---

#### Model Predictions and Evaluation Labels

Background context: In machine learning, model predictions often need to be evaluated against some ground truth. This is typically done by logging the predictions alongside actual labels, which can then be analyzed using various tools like Grafana, ELK, or Prometheus.

:p How are evaluation labels derived from model predictions?
??x
Evaluation labels are derived by logging the predictions made by a model along with their corresponding true outcomes (relevance scores or actual labels). This log data is then processed using log-parsing technologies to extract meaningful insights and metrics. 
```java
// Example of logging prediction in Java
public void logPrediction(double predictedScore, boolean actualRelevance) {
    logger.log("Prediction: " + predictedScore + ", Actual Relevance: " + actualRelevance);
}
```
x??

---

#### Receiver Operating Characteristic (ROC) Curve

Background context: The ROC curve is a graphical representation of the performance of a binary classifier system as its discrimination threshold is varied. It plots the true positive rate against the false positive rate at various threshold settings.

:p What does an ROC curve help us estimate in the context of recommendation systems?
??x
An ROC curve helps estimate how well the relevance scores predict whether an item will be relevant to a user, by plotting the true positive rate (TPR) against the false positive rate (FPR). This can inform necessary retrieval depth and identify problematic queries.
```java
// Pseudocode for calculating TPR and FPR for ROC curve
public class RocCurve {
    public double calculateTpr(double[] actualScores, double[] predictedScores) {
        // Logic to compute true positive rate
        return tpr;
    }
    
    public double calculateFpr(double[] actualScores, double[] predictedScores) {
        // Logic to compute false positive rate
        return fpr;
    }
}
```
x??

---

#### Continuous Training and Deployment

Background context: Models in production often require continuous updates due to changing data distributions or performance degradations. This involves monitoring model performance, detecting drift, and retraining the models with new data.

:p What is model drift?
??x
Model drift refers to a scenario where a model's behavior changes over time because of shifts in the underlying data distribution. For instance, a model that performs well on historical data may degrade if trained on outdated or different data.
```java
// Example code snippet for detecting model drift using sequential cross-validation
public class DriftDetector {
    public void trainAndEvaluate(double[] oldData, double[] newData) {
        // Train and evaluate the model on the new data with a time delay
    }
}
```
x??

---

#### Ensemble Modeling

Background context: Ensembles combine predictions from multiple models to improve overall performance. The mixture of expert opinions often outperforms a single estimator.

:p What is an ensemble in machine learning?
??x
An ensemble in machine learning refers to a method where multiple models are built, and their predictions are combined (e.g., through averaging or voting) to produce the final output. This approach can smooth out problematic predictions and provide more robust performance.
```java
// Example of implementing bagging for ensemble modeling
public class EnsembleModel {
    public double[] predict(double[] inputs) {
        // Bagging implementation
        return combinedPredictions;
    }
}
```
x??

---

#### Shadowing

Background context: Shadowing involves deploying two models simultaneously to compare their performance before making the new model live. This helps ensure that predictions from the new model align with expectations.

:p What is shadowing in the context of model deployment?
??x
Shadowing refers to a technique where two models, even for the same task, are deployed side by side. One model handles production traffic while the other processes requests silently and logs its results. This allows evaluating the performance of new models before making them live.
```java
// Example code snippet for implementing shadowing
public class ShadowModel {
    public void processRequest(double[] input) {
        // Process request in a silent mode, logging results
    }
}
```
x??

---

#### Experimentation

Background context: Proper experimentation frameworks are crucial for validating the performance of new models. Techniques like A/B testing and multi-armed bandit algorithms can be used to compare different models.

:p What is the role of experimentation in model deployment?
??x
Experimentation plays a critical role in validating the performance of new models. It involves deploying multiple models simultaneously, using techniques like A/B testing or multi-armed bandits, where the controller layer decides which model's response to send based on predefined rules.
```java
// Example code snippet for simple A/B experimentation
public class ExperimentController {
    public void routeRequest(double[] input) {
        // Randomly decide which model to use for the request
    }
}
```
x??

---

#### Model Cascades

Background context: Model cascades extend the concept of ensembling by using confidence estimates. If a model's prediction is uncertain, it passes the task to another downstream model.

:p What is a model cascade?
??x
A model cascade involves using confidence estimates from initial models to decide whether to return their predictions or pass tasks to subsequent models in the ensemble. This approach allows for iterative refinement of predictions by leveraging multiple models.
```java
// Example code snippet for implementing a simple model cascade
public class ModelCascade {
    public double predict(double[] input) {
        // Check confidence and call downstream model if needed
        return finalPrediction;
    }
}
```
x??

#### Daily Warm Starts
Background context: Daily warm starts involve updating models incrementally using new data without a full retraining. This approach is particularly useful for large recommendation models where full retraining would be computationally expensive and time-consuming.

:p What is daily warm starts used for?
??x
Daily warm starts are used to update the model with new data seen each day, avoiding full retraining which can be resource-intensive and time-consuming.
x??

---

#### Lambda Architecture and Orchestration
Background context: The lambda architecture aims to handle real-time streaming data by combining batch processing with a speed layer. This ensures that both historical data and real-time updates are processed efficiently.

:p What is the purpose of the lambda architecture?
??x
The purpose of the lambda architecture is to ensure efficient processing of both batch and stream data, combining historical data with real-time updates.
x??

---

#### Evaluation Flywheel
Background context: The evaluation flywheel describes a feedback loop mechanism where models are continuously improved based on new data. This includes retraining, logging, and deployment to maintain model performance over time.

:p What is the evaluation flywheel?
??x
The evaluation flywheel refers to a continuous improvement process for machine learning models, involving regular retraining with new data, detailed logging, and deployment updates.
x??

---

#### Collector Logs
Background context: Logging in the collector layer involves tracking requests from users to the recommendation system. This helps ensure that the system is functioning correctly and allows for troubleshooting.

:p What are the key steps involved in collector logs?
??x
Key steps in collector logs include logging when receiving a request, looking up embeddings, computing approximate nearest neighbors (ANN), applying filters, augmenting features, scoring candidates, and ordering recommendations.
x??

---

#### Filtering and Scoring Logs
Background context: Detailed logging during filtering and scoring helps maintain transparency and traceability of the recommendation process. This ensures that each step can be audited for correctness.

:p What should be logged during filtering and scoring?
??x
During filtering and scoring, logs should capture incoming requests to the filtering service, filter applications, bloom filters used, feature augmentation from the feature store, scoring candidates with the ranking model, and any potential confidence estimations.
x??

---

#### Ordering and Application of Business Logic or Experimentation Logs
Background context: The final step in the recommendation pipeline involves applying business logic and running experiments. Detailed logging ensures that the rationale behind each decision is clear for future reference.

:p What should be logged during the ordering step?
??x
During the ordering step, logs should capture incoming candidates, reasons for elimination, application of business rules, experiment IDs, and the state of the recommendation before finalizing it.
x??

---

#### Structured Logs and Log Formatting
Background context: Using structured logs with a log-formatter object helps in parsing and writing logs efficiently. This ensures that there is a tight coupling between logs and application logic.

:p Why are structured logs important?
??x
Structured logs are important because they facilitate easier parsing, writing, and integration with application logic, providing clear and detailed records of system behavior.
x??

---

#### Tight Coupling between Objects and Logs

Background context: In service architectures, tight coupling between objects of execution and their logs can simplify maintenance and reduce overhead. When changes are made to the application code, these changes can be reflected in the logs without additional steps, making it easier to track and understand the flow of operations.

:p How does using the same objects for both your service logic and logging save time and effort?
??x
Using the same objects for both execution and logging means that when you update or modify your application code, the changes propagate automatically to the logs. This reduces the need for additional steps where you would otherwise have to ensure that log messages reflect those changes manually. By using a consistent object model across these components, testing can be streamlined because any modifications in the objects will naturally appear in the logs due to the use of the same objects during execution.

For example, if your service uses `User` objects for handling user data and you also configure a logging mechanism that formats log entries with the same `User` class, then updating or modifying the `User` class will automatically reflect these changes in the generated logs. This can be particularly useful when developing unit tests where you want to ensure that specific fields of your `User` objects are visible and correctly formatted in the logs.

```java
// Example Java code for a User object and logging mechanism
public class User {
    private String name;
    
    public User(String name) {
        this.name = name;
    }
}

public class LoggingService {
    private final Logger logger = LoggerFactory.getLogger(LoggingService.class);
    
    public void logUser(User user) {
        // Assuming the user's details are logged using a formatter that uses the same User object
        String formattedLogMessage = "User: " + user.getName();
        logger.info(formattedLogMessage, user);
    }
}
```

x??

---

#### Active Learning in Recommendation Systems

Background context: Active learning is an approach where the model actively selects data points to label or prioritize for annotation. This can be particularly useful in recommendation systems (RecSys) where cold-starting new items and broadening users' interests are key challenges.

:p What is active learning, and why might it be important in a RecSys?
??x
Active learning is a method of semi-supervised machine learning where the model selects the most informative data points to label. In recommendation systems (RecSys), this can help address issues like the Matthew effect, which occurs when popular items receive more ratings while less popular ones are starved for attention.

By using active learning, the system can prioritize showing new or potentially good matches to users who haven't rated them yet, thereby generating data that can improve the performance of the recommendation model over time. For example, a newly added movie could be recommended to the first 100 customers as a way to gather initial ratings and start the cold-start process.

```java
// Example Java code for active learning in a RecSys context
public class Recommender {
    private final Model model;
    
    public Recommender(Model model) {
        this.model = model;
    }
    
    public void selectActiveSamples() {
        // Logic to select items that would provide the most value when labeled
        List<Item> selectedItems = model.getActiveSamples();
        
        for (Item item : selectedItems) {
            // Recommend the selected items to users and wait for ratings
            recommend(item);
        }
    }
    
    private void recommend(Item item) {
        // Code to recommend the item to users
    }
}
```

x??

---

#### Nonpersonalized vs. Personalized Active Learning

Background context: In nonpersonalized active learning, the goal is to minimize loss over the entire system rather than just for a single user. Personalized active learning focuses on improving performance and data collection for specific users based on their historical interactions.

:p What are the differences between personalized and nonpersonalized active learning in RecSys?
??x
In recommendation systems (RecSys), the main difference between personalized and nonpersonalized active learning lies in the focus of the optimization process. Nonpersonalized active learning aims to minimize loss over the entire system, considering the collective performance rather than individual users' preferences.

Personalized active learning, on the other hand, tailors the selection of data points for labeling based on a user's specific history and behavior. This approach uses detailed information about each user to make more targeted and relevant decisions.

For example:
- **Nonpersonalized Active Learning** might focus on selecting items that have high variance in user ratings across the board, hoping to gather diverse feedback.
- **Personalized Active Learning** would prioritize recommending new content to users who show signs of interest but haven't rated certain items yet, aiming to maximize engagement and data collection from those specific users.

```java
// Example Java code for nonpersonalized active learning in a RecSys context
public class Recommender {
    private final Model model;
    
    public Recommender(Model model) {
        this.model = model;
    }
    
    public void selectNonPersonalSamples() {
        // Logic to select items that have high variance in user ratings
        List<Item> selectedItems = model.getHighVarianceItems();
        
        for (Item item : selectedItems) {
            recommend(item);
        }
    }
    
    private void recommend(Item item) {
        // Code to recommend the item to users and wait for ratings
    }
}
```

x??

---

