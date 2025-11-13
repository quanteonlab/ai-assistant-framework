# Flashcards: 2A014 (Part 23)

**Starting Chapter:** 96-Recommendation Probabilities to AUC-ROC

---

#### Relevance of Individual Items and Positioning

Background context explaining the concept. In recommendation systems, metrics like mAP (Mean Average Precision), MRR (Mean Reciprocal Rank), and NDCG (Normalized Discounted Cumulative Gain) are crucial because they consider both the relevance of individual items and their positions in the ranking. A high correlation coefficient might indicate a good linear relationship but does not necessarily reflect accurate rankings.

:p What aspect do mAP, MRR, and NDCG focus on that a correlation coefficient might miss?

??x
mAP, MRR, and NDCG focus on the relevance of individual items and their positions in the ranking. A high correlation coefficient can indicate a good linear relationship but does not capture nuances such as item importance or correct item order.

For example:
- If a recommender system correctly identifies the right items but ranks them incorrectly (e.g., important items are ranked low), it might have a high correlation coefficient but poor performance in mAP, MRR, and NDCG.
x??

---

#### Relevance and Positioning Impact on Ranking Metrics

Background context explaining the concept. When a recommender system predicts that a user will interact with certain items again, the order in which these items are presented can significantly impact metrics like mAP, MRR, and NDCG. Even if the correct items are predicted, an incorrect order can lead to poor ranking performance.

:p How does the position of relevant items affect the performance metrics?

??x
The position of relevant items affects the performance metrics such as mAP, MRR, and NDCG because these metrics consider not just whether the items are included in the top ranks but also their relative positions. For example, if a user's most important item is ranked low (e.g., 10th out of 20), this negatively impacts all three metrics.

For instance:
- mAP: Affects by how well the system ranks relevant items.
- MRR: Sees a significant drop if highly relevant items are not at the top.
- NDCG: Penalizes systems that rank important items poorly, even if they predict them correctly.

Example in code (pseudocode):
```java
public class RankingEvaluation {
    public double evaluate(List<Item> predictedItems, List<Item> trueItems) {
        double mAP = calculateMeanAveragePrecision(predictedItems, trueItems);
        double MRR = calculateMeanReciprocalRank(predictedItems, trueItems);
        double NDCG = calculateNormalizedDiscountedCumulativeGain(predictedItems, trueItems);
        return mAP + MRR + NDCG;
    }
}
```
x??

---

#### Correlation Coefficient vs. Ranking Metrics

Background context explaining the concept. While correlation coefficients can provide a high-level understanding of ranking performance by measuring linear relationships between predicted and actual values, they are not sufficient for evaluating recommendation systems where the order of items is crucial.

:p Why might a correlation coefficient be insufficient in evaluating a recommender system?

??x
A correlation coefficient may be insufficient because it measures only the strength and direction of a linear relationship between two variables without considering their relative positions or the relevance of individual items. In recommendation systems, the order and relevance are critical, whereas a correlation coefficient does not account for these nuances.

For example:
- A system that correctly predicts user interactions but ranks important items poorly might have a high correlation coefficient due to its overall predictive ability but poor performance in ranking metrics like mAP, MRR, and NDCG.
x??

---

#### Root Mean Square Error (RMSE) vs. Ranking Metrics

Background context explaining the concept. RMSE is a metric used for regression tasks, measuring the average squared difference between predicted affinity scores and true values. However, it does not consider the ranking structure of recommendation systems.

:p How do RMSE and ranking metrics like mAP, MRR, and NDCG differ in their evaluation approach?

??x
RMSE measures the accuracy of predicting affinity scores by calculating the square root of the average squared differences between predicted and actual values. It is a regression metric that does not consider the order or relevance of items.

In contrast, mAP, MRR, and NDCG are designed specifically for evaluating ranking quality in recommendation systems. They take into account both the presence and position of relevant items.

For instance:
- RMSE: `sqrt(mean((predicted - actual)^2))`
- mAP: Average precision at each relevant item.
- MRR: Reciprocal rank of the first relevant item.
- NDCG: Discounted cumulative gain, penalizing low ranks for important items.

Example in code (pseudocode):
```java
public class EvaluationMetrics {
    public double rmse(List<Double> predictions, List<Double> actuals) {
        return Math.sqrt(meanSquaredError(predictions, actuals));
    }

    public double map(List<Item>, List<Item>) {
        // Calculate mAP
    }

    public double mrr(List<Item>, List<Item>) {
        // Calculate MRR
    }

    public double ndcg(List<Item>, List<Item>) {
        // Calculate NDCG
    }
}
```
x??

---

#### Area Under the Curve (AUC) and cAUC

Background context explaining the concept. AUC and cAUC are used to evaluate ranking quality by considering how well a model ranks positive examples over negative ones.

:p What do AUC and cAUC measure in recommendation systems?

??x
AUC measures the probability that a randomly chosen positive instance is ranked higher than a randomly chosen negative instance. cAUC (Conditional AUC) extends this to consider different conditions or subsets of data, providing more granular insights into ranking quality.

For example:
- AUC: `P(X_positive > X_negative)` where `X` represents affinity scores.
- cAUC considers subgroups and their rankings, such as users with specific attributes.

Example in code (pseudocode):
```java
public class AucEvaluation {
    public double auc(List<Item> positives, List<Item> negatives) {
        // Calculate AUC
    }

    public double cAuc(List<Item> positives, List<Item> negatives, String condition) {
        // Calculate cAUC for a specific condition
    }
}
```
x??

#### AUC-ROC (Area Under the Receiver Operating Characteristic Curve)
Background context: In a binary classification setup, AUC-ROC measures the ability of the recommendation model to distinguish between positive (relevant) and negative (irrelevant) instances. It is calculated by plotting the true positive rate (TPR) against the false positive rate (FPR) at various threshold settings and then computing the area under this curve.

:p What does AUC-ROC measure in a recommendation system?
??x
AUC-ROC measures how well the model ranks relevant items over irrelevant ones, regardless of their actual rank position. It effectively quantifies the likelihood that a randomly chosen relevant item is ranked higher than a randomly chosen irrelevant one by the model.
x??

---

#### Recommendation Probabilities and AUC-ROC
Background context: In the context of recommendations, "thresholds" can be thought of as varying the number of top items recommended to a user. The affinity score represents a confidence measure by the model that an item is relevant.

:p How does AUC-ROC relate to recommendation probabilities in terms of ranking?
??x
AUC-ROC assesses how well your model ranks relevant items over irrelevant ones irrespective of their exact rank position. It considers the relative ranking rather than absolute positions, making it useful for understanding the overall effectiveness of item recommendations.
x??

---

#### mAP (Mean Average Precision)
Background context: This metric averages precision values computed at the ranks where each relevant item is found. Unlike AUC-ROC, which does not account for position bias and considers all relevant items in the list, mAP provides a more nuanced evaluation by emphasizing higher-ranked items.

:p What does mAP measure in recommendation systems?
??x
mAP measures model performance by averaging precision values at the ranks where each relevant item is found. It focuses on the top rankings and thus is more sensitive to changes near the top of the list.
x??

---

#### MRR (Mean Reciprocal Rank)
Background context: Unlike AUC-ROC and mAP, which consider all relevant items in the list, MRR focuses only on the rank of the first relevant item. It measures how quickly the model can find a relevant item.

:p What does MRR measure in recommendation systems?
??x
MRR measures the quality of ranking by focusing solely on the position of the first relevant item in the list. A higher MRR indicates that the model consistently places relevant items at the top.
x??

---

#### NDCG (Normalized Discounted Cumulative Gain)
Background context: This metric evaluates not only the order of recommendations but also takes into account the graded relevance of items, discounting items further down the list.

:p What does NDCG measure in recommendation systems?
??x
NDCG measures the quality of ranking by considering both the order and the graded relevance of recommended items. It rewards relevant items that appear higher up in the list more than those lower down.
x??

---

#### cAUC (Customer AUC)
Background context: Sometimes, AUC is computed per customer and then averaged to provide a better expectation for individual user experiences.

:p What does cAUC measure?
??x
cAUC measures the AUC score per customer and averages it across customers. This provides a personalized evaluation of the model’s performance based on each user's experience.
x??

---

#### BPR (Bayesian Personalized Ranking)
Background context: BPR presents a Bayesian approach to item ranking in recommendation systems, focusing on pairwise preferences rather than binary classification.

:p What does BPR do differently from other metrics discussed?
??x
BPR focuses on pairwise preferences by comparing two items for a specific user and optimizing the posterior probability of observed rankings being correct. Unlike AUC-ROC, mAP, MRR, and NDCG, which evaluate performance post-training, BPR guides model learning directly toward optimizing ranking.
x??

---

#### Ranking in Recommender Systems
Background context explaining how ranking fits into larger recommender systems. Mention the two-phase process: retrieval followed by ranking, and explain why more expensive models can be used during ranking due to smaller candidate sets.

:p Where does ranking fit within a typical large-scale recommender system?
??x
Ranking is an integral part of large-scale recommender systems, following the retrieval phase where a quick function gathers a set of candidate items. After retrieving these candidates, ranking helps in ordering them based on their relevance. The candidate set is usually smaller than the entire item corpus, allowing for more complex and expensive models to be applied during this phase. Features like user features, context features, and item representations are concatenated to create feature vectors used for scoring and ranking.
x??

---

#### Types of Learning to Rank (LTR)
Explanation of different LTR approaches: pointwise, pairwise, and listwise, including their goals.

:p What are the three main types of learning to rank models?
??x
There are three main types of learning to rank (LTR) models:
1. **Pointwise**: The model treats individual documents in isolation and assigns them a score or rank. This is essentially treating the problem as a regression or classification task.
2. **Pairwise**: The model considers pairs of documents simultaneously in the loss function, aiming to minimize the number of incorrectly ordered pairs.
3. **Listwise**: The model considers the entire list of documents in the loss function and tries to find the optimal ordering for the entire list.

These approaches differ in how they handle the order and relevance of items during training.
x??

---

#### Pointwise Learning to Rank
Explanation of pointwise LTR, focusing on treating each document individually and converting it into a regression or classification problem.

:p What is the goal of pointwise learning to rank?
??x
The goal of pointwise learning to rank is to assign scores or ranks to individual documents in isolation. This approach treats the ranking task as either a regression (predicting continuous scores) or classification (predicting discrete relevance labels) problem, focusing on optimizing the score for each document independently.
x??

---

#### Pairwise Learning to Rank
Explanation of pairwise LTR, which focuses on comparing pairs of items and minimizing incorrectly ordered pairs.

:p What does the pairwise approach in learning to rank do?
??x
The pairwise approach in learning to rank considers pairs of documents simultaneously in the loss function. The objective is to minimize the number of incorrectly ordered pairs by ensuring that for any given pair $(i, j)$, if item $ i$should be ranked higher than item $ j$, then its score must be greater than or equal to the score of $ j$.
x??

---

#### Listwise Learning to Rank
Explanation of listwise LTR, which considers the entire list in the loss function and aims for the optimal ordering.

:p What is the goal of listwise learning to rank?
??x
The goal of listwise learning to rank is to find the optimal ordering of the entire list. This approach directly optimizes the order of items as a whole rather than focusing on individual scores or pairs, making it suitable for tasks where the overall structure of the ranked list matters.
x??

---

