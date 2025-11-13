# Flashcards: 2A014 (Part 21)

**Starting Chapter:** 89-User Versus Item Metrics

---

#### Evaluation Types for Recommendation Systems

Background context explaining that different types of evaluations (online/offline, user/item, A/B) provide unique insights into how well a recommendation system performs. These methods help ensure the recommendations are relevant and align with user preferences.

:p What are the different evaluation setups mentioned in the text?
??x
The different evaluation setups include online/offline, user/item, and A/B testing.
x??

---

#### Online vs Offline Evaluation

Explanation of online versus offline evaluation, where offline evaluations use a test/evaluation dataset outside the production system to compute metrics. This approach leverages historical data but requires sufficient existing data.

:p What is an offline evaluation?
??x
Offline evaluation involves using a test/evaluation dataset outside the production system to compute a set of metrics. It relies on historical data to simulate inference and construct relevant responses.
x??

---

#### Prequential Data

Explanation of prequential data, which is more relevant in recommendation systems than in other ML applications due to its sequential nature, focusing on historical exposure.

:p What is prequential data?
??x
Prequential data refers to a dataset that is used in the context of large models and recommendation systems. It emphasizes historical exposure and is crucial for sequential recommenders.
x??

---

#### Sequential Recommendation

Explanation of why people often say "all recommenders are sequential" due to the importance of historical exposure.

:p Why do all recommenders being sequential matter?
??x
The statement that "all recommenders are sequential" highlights the critical role of historical exposure in recommendation systems. This is because user preferences and behaviors evolve over time, making historical data essential for accurate and personalized recommendations.
x??

---

#### RecList Evaluation Framework

Explanation of the RecList project which builds a checklist-based framework to organize metrics and evaluations for recommender systems.

:p What is RecList?
??x
RecList is a project that provides a useful checklist-based framework for organizing metrics and evaluations in recommendation systems, offering a comprehensive view on how to assess the performance of these systems.
x??

---

#### Online Evaluation vs. Offline Metrics
Online evaluation takes place during inference, usually in production. It involves computing metrics like frequency and distributions of covariates, CTR/success rate, or time on platform. However, these are different from offline metrics which are typically used for training purposes.

:p What is the difference between online evaluation and offline metrics?
??x
Online evaluation occurs during real-time inference in production to assess model performance using live data. It focuses on practical metrics such as frequency of recommendations, success rates (CTR), and user engagement time. Offline metrics, on the other hand, are computed from historical data used for training and validation. These include precision, recall, AUC-ROC, etc., which are not directly observed in production but provide a theoretical benchmark.

```java
// Example code to log online evaluation metrics
public class OnlineEvaluator {
    private long totalImpressions = 0;
    private int successfulClicks = 0;

    public void logImpression() { totalImpressions++; }
    public void logClick() { successfulClicks++; }

    public double getCTR() { return (double)successfulClicks / totalImpressions; }
}
```
x??

---

#### Bootstrapping from Historical Evaluation Data
Bootstrapping involves using historical data to build a recommender system, especially when no initial user interactions are available. This can be achieved by asking users for preference information or using co-occurrence data.

:p How do you start building a recommender system with limited training data?
??x
You can bootstrap your recommender system by leveraging existing co-occurrence data or querying users for their preferences. For instance, in the Wikipedia example, co-occurrences between articles were used without needing user interactions. Alternatively, you could ask users to rate or rank items, which helps build initial preference models.

```java
// Example of bootstrapping using item ratings
public class BootstrapRecommender {
    private Map<String, List<Double>> userRatings = new HashMap<>();

    public void addUserRating(String userId, String itemId, double rating) {
        if (!userRatings.containsKey(userId)) {
            userRatings.put(userId, new ArrayList<>());
        }
        userRatings.get(userId).add(rating);
    }

    // Further processing to build recommendation model
}
```
x??

---

#### User versus Item Metrics
In recommender systems, both user and item metrics are important. User metrics measure the performance of recommendations from a user perspective, while item metrics assess how frequently items get recommended.

:p Why is it important to consider both user and item metrics in recommenders?
??x
It's crucial to evaluate both user and item metrics because focusing solely on user satisfaction might neglect the importance of promoting less popular but still valuable items. User metrics like click-through rate (CTR) help understand how well recommendations align with users' preferences, while item metrics ensure that all relevant items get a fair chance to be recommended.

```java
// Example code for computing CTR and item frequency
public class RecommenderMetrics {
    private Map<String, Integer> itemRecommendations = new HashMap<>();
    private int totalUserInteractions = 0;

    public void logInteraction(String itemId) {
        totalUserInteractions++;
        if (itemRecommendations.containsKey(itemId)) {
            itemRecommendations.put(itemId, itemRecommendations.get(itemId) + 1);
        } else {
            itemRecommendations.put(itemId, 1);
        }
    }

    public double getCTR() { return (double)totalUserInteractions / (itemRecommendations.size()); }
    public Map<String, Integer> getItemFrequencies() { return Collections.unmodifiableMap(itemRecommendations); }
}
```
x??

---

#### A/B Testing for Recommenders
A/B testing involves deploying two or more recommender models to measure their performance. It helps in estimating the effect size of model changes and ensuring that new models perform better than existing ones.

:p How does A/B testing work for recommender systems?
??x
A/B testing in recommenders typically deploys multiple models to a subset of users and compares their performance metrics over time. This helps estimate the causal impact of new models on user behavior, such as CTR or engagement. The randomization unit can be at the user level but should consider potential covariates like seasonal effects.

```java
// Example A/B test setup
public class ABTestSetup {
    private Map<String, String> userToModel = new HashMap<>();

    public void assignUser(String userId, String modelVersion) { userToModel.put(userId, modelVersion); }

    public String getUserModel(String userId) { return userToModel.get(userId); }
}

// Usage example
ABTestSetup setup = new ABTestSetup();
setup.assignUser("user123", "modelA");
String userModel = setup.getUserModel("user123"); // Returns the assigned model version
```
x??

---

#### Recall and Precision in Recommenders
Recall measures how many relevant items are included in the recommended list, while precision measures the proportion of recommended items that are actually relevant.

:p How do recall and precision differ in recommender systems?
??x
In recommender systems, **recall** is about ensuring that relevant items are included in the recommendations. It's particularly important when the number of potential relevant results is small compared to the total number of recommendations. On the other hand, **precision** focuses on how many of the recommended items are actually relevant, useful for scenarios where there are many irrelevant recommendations.

```java
// Example calculation of recall and precision
public class RecommenderEvaluation {
    private Set<String> relevantItems = new HashSet<>();
    private List<String> recommendedItems = new ArrayList<>();

    public void addRelevantItem(String itemId) { relevantItems.add(itemId); }
    public void addRecommendedItem(String itemId) { recommendedItems.add(itemId); }

    public double getRecall() { return (double)relevantItems.size() / recommendedItems.size(); }
    public double getPrecision() { return (double)relevantItems.retainAll(recommendedItems) / relevantItems.size(); }
}
```
x??

#### Intersection of Recommendation and Relevance
Background context: The intersection between recommendation and relevance can vary significantly, often being small or even empty. This variability affects how many recommendations match relevant options.

:p How does the size of the intersection between recommendation and relevance impact the recommender system's performance?
??x
The size of the intersection directly influences the precision and recall metrics of the recommender system. A smaller intersection typically means lower precision (the fraction of recommended items that are actually relevant) and recall (the fraction of relevant items that are correctly identified by recommendations). If the intersection is empty, it indicates no overlap between the recommended items and known relevant options.
x??

---

#### Precision and Recall @ k
Background context: Precision and recall metrics evaluate the quality of recommendations at a specific number `k` of top-ranked items. These metrics help understand how well the system ranks relevant items among its top suggestions.

:p What do precision and recall @ k measure in a recommender system?
??x
Precision @ k measures the fraction of recommended items that are actually relevant out of all the recommended items. Recall @ k measures the fraction of relevant items correctly identified by the recommendations out of all relevant items.
```java
// Example pseudocode for calculating precision and recall at k
public class RecommendationEvaluator {
    public double calculatePrecision(int[] relevanceVector, int k) {
        Set<Integer> topKRecoms = new HashSet<>();
        for (int i = 0; i < k; i++) {
            topKRecoms.add(relevanceVector[i]);
        }
        
        int relevantInTopK = 0;
        for (Integer item : topKRecoms) {
            if (relevanceVector[item] == 1) { // assuming relevance is indicated by 1
                relevantInTopK++;
            }
        }
        
        return (double) relevantInTopK / k; // Precision @ k
    }

    public double calculateRecall(int[] relevanceVector, int r, int k) {
        Set<Integer> topKRecoms = new HashSet<>();
        for (int i = 0; i < k; i++) {
            topKRecoms.add(relevanceVector[i]);
        }
        
        int relevantInTopK = 0;
        for (Integer item : relevanceVector) {
            if (item == 1 && topKRecoms.contains(item)) { // assuming relevance is indicated by 1
                relevantInTopK++;
            }
        }
        
        return (double) relevantInTopK / r; // Recall @ k
    }
}
```
x??

---

#### Cardinality of Relevant Items (@r)
Background context: The cardinality of the set of relevant items, denoted as `@r`, is crucial for calculating recall. It represents the total number of known relevant options available in the training or test data.

:p How does the cardinality of relevant items (@r) affect the calculation of recall?
??x
The cardinality of relevant items (`@r`) affects the denominator when calculating recall, which measures how well the system captures all relevant items. The larger `@r`, the harder it is to achieve a high recall value if not enough relevant items are included in the top-k recommendations.

For example:
```java
// Example calculation of Recall with @r known
public class RecallCalculator {
    public double calculateRecall(int[] relevanceVector, int r) {
        int relevantInTopK = 0;
        for (int item : relevanceVector) {
            if (item == 1) { // assuming relevance is indicated by 1
                relevantInTopK++;
            }
        }
        
        return (double) relevantInTopK / r; // Recall @ k, where k is the length of the relevanceVector
    }
}
```
x??

---

#### Difference Between Precision and Recall
Background context: Precision focuses on the quality of recommendations by ensuring that most recommended items are indeed relevant. On the other hand, recall measures how well the system covers all relevant items.

:p What distinguishes precision from recall in a recommender system?
??x
Precision in a recommender system is defined as the fraction of recommended items that are actually relevant out of all the recommended items:
$$\text{Precision} = \frac{\text{Number of true positives}}{\text{Number of true positives + Number of false positives}}$$

Recall, on the other hand, measures the fraction of relevant items that are correctly identified by the recommendations out of all relevant items:
$$\text{Recall} = \frac{\text{Number of true positives}}{\text{Number of true positives + Number of false negatives}}$$

Precision and recall are complementary metrics; improving one may come at the cost of the other. A high precision means fewer irrelevant items, but it might miss some relevant ones (low recall), while a high recall ensures more relevant items are captured, potentially at the expense of including irrelevant ones.
x??

---

