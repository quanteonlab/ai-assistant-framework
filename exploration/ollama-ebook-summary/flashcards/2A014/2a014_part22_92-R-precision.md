# Flashcards: 2A014 (Part 22)

**Starting Chapter:** 92-R-precision

---

---
#### Precision at k
Background context: Precision is a measure of the accuracy of the recommendations made by a recommendation system. It calculates how many of the recommended items are relevant to the user.

The formula for precision at $k$ is:
$$Precision @k = \frac{\text{numrelevant}}{k}$$where `numrelevant` is the number of relevant items in the top $ k $ recommendations, and $ k$ is the total number of recommended items.

:p What does the Precision at k metric measure?
??x
The precision at $k $ measures the accuracy of the recommendation system by calculating the ratio of relevant items among the first$k$ recommendations. It helps understand how many of the top suggested items are actually useful or desired by the user.
x??

---
#### Recall at k
Background context: Recall is another metric used to evaluate a recommendation system, focusing on the proportion of relevant items that were successfully recommended.

The formula for recall at $k$ can be expressed as:
$$Recall @k = \frac{\text{numrelevant}}{\text{max}(r, k)}$$where `numrelevant` is the number of relevant items in the top $ k $ recommendations, and $ r$ is the total number of relevant items that exist. The `max(r, k)` ensures that recall considers the maximum possible size of relevant items.

:p How is the Recall at k metric calculated?
??x
The recall at $k $ metric calculates how well the recommendation system covers all relevant items by considering the ratio of relevant recommendations to either the total number of relevant items or the top$k $ recommended items, whichever is larger. This ensures that if$r > k$, the denominator still reflects the actual count of relevant items.
x??

---
#### Scenario 3: Streaming Platform Recall
Background context: In a scenario where users are looking for specific content on a streaming platform, recall measures how well the system suggests the user's desired movies or shows.

The core concept here is:
- The number of relevant movies (desired by the user) that appear in the recommendation list.
- The total count of all available media items.

If the entire set of desired media items is found on this platform, it indicates a recall of 100%.

:p What does the scenario illustrate about Recall?
??x
The scenario illustrates how recall measures the effectiveness of a recommendation system by checking if relevant content (desired movies or shows) are included in the recommended list. A high recall value means that most desired items are found among the recommendations, which is crucial for user satisfaction.
x??

---
#### Scenario 4: Café Experience and Avoid Recall
Background context: In scenarios where users have a wide range of preferences, the concept of "avoid" can be useful to measure how well a recommendation system performs. Instead of focusing on what users like, it looks at what they do not want.

The formula for avoid recall is:
$$Avoid @k = \frac{\text{numrelevant}}{k - \text{numrelevant}}$$and the recall adjusted by this factor is:
$$

Recall @k = k - Avoid @k$$:p How does Scenario 4 use Recall and Avoid to evaluate recommendations?
??x
Scenario 4 uses a different approach to recall, focusing on what users do not want rather than what they like. By calculating avoid recall, it measures the number of irrelevant items among the top $k$ recommendations. The adjusted recall then reflects how well the system avoids recommending unwanted items. This provides an alternative way to evaluate recommendation systems in scenarios where users have a large set of dislikes.
x??

---

#### mAP (Mean Average Precision)
Background context explaining the concept. The mAP metric evaluates recommendation systems by considering both the relevance and the ranking of recommended items. It computes the average precision for each query at various cutoffs, providing a comprehensive measure.

Formula:
$$\text{mAP} = \frac{1}{Q} \sum_{q=1}^{Q} \frac{1}{m_q} \sum_{k=1}^{n_q} P_k \cdot rel_k$$

Where $Q $ is the total number of queries,$m_q $ is the number of relevant documents for a specific query$q $, and$ P_k $stands for precision at the$ k $-th cutoff. The indicator function $ rel_k = 1$ if the item at rank $ k $ is relevant; otherwise, $ rel_k = 0$.

:p What does mAP measure in a recommendation system?
??x
mAP measures the average precision across all queries, taking into account both the relevance and the ranking of the top-recommended items. It provides an overall assessment by averaging the precision scores at various cutoffs.
x??

---

#### MRR (Mean Reciprocal Rank)
Background context explaining the concept. The MRR metric focuses on the position of the first relevant item in a recommendation list, giving it more weight when the relevant item is ranked higher.

Formula:
$$\text{MRR} = \frac{1}{Q} \sum_{i=1}^{Q} \frac{1}{rank_i}$$

Where $Q $ represents the total number of queries, and$rank_i $ is the position of the first relevant item in the list for query$i$.

:p What does MRR measure in a recommendation system?
??x
MRR measures the average reciprocal rank of the first relevant item across all queries. It emphasizes higher ranks of the first relevant items, providing an insight into how quickly the algorithm identifies relevant recommendations.
x??

---

#### NDCG (Normalized Discounted Cumulative Gain)
Background context explaining the concept. The NDCG metric evaluates recommendation systems by considering the ranking and relevance of items, but it discounts the relevance as we move further down the list.

Formula:
$$\text{NDCG} = \frac{\text{DCG}}{\text{IDCG}}$$

Where DCG (Discounted Cumulative Gain) is calculated as:
$$\text{DCG} = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}$$

And IDCG (Ideal Discounted Cumulative Gain) is the optimal DCG value when all relevant items are at the top of the list.

Formula for NDCG@k:
$$\text{NDCG}_{@k} = \frac{\sum_{i=1}^{k} \frac{rel_i}{\log_2(i+1)}}{\sum_{i=1}^{|ℛ|} \frac{rel_i}{\log_2(i+1)}}$$

Where $ℛ$ is the set of relevant documents.

:p What does NDCG measure in a recommendation system?
??x
NDCG measures how well the recommendation algorithm ranks relevant items, discounting their relevance as they appear further down the list. It provides a normalized score that accounts for both ranking and relevance.
x??

---

#### mAP Versus NDCG

Background context: Both mAP (Mean Average Precision) and NDCG (Normalized Discounted Cumulative Gain) are comprehensive metrics for evaluating ranking quality, considering all relevant items and their respective ranks. While both provide a holistic view of recommendation systems, they offer different insights and use cases.

mAP is the average precision at each relevant item across different relevance thresholds. It effectively represents the area under the precision-recall curve. On the other hand, NDCG considers the relevance of each item with a logarithmic discount factor to quantify its importance based on its position in the list.

:p What are the key differences between mAP and NDCG?
??x
mAP focuses on the average precision at different relevant items, providing an intuitive understanding of the trade-off between precision and recall. NDCG, however, discounts the relevance of items based on their rank using a logarithmic factor, making it more sensitive to the order of items.

```java
public class Example {
    public double calculateMAP(double[] relevances) {
        // Implementation for calculating mAP
    }
    
    public double calculateNDCG(double[] relevances, int k) {
        // Implementation for calculating NDCG up to rank k
    }
}
```
x??

---

#### MRR (Mean Reciprocal Rank)

Background context: MRR does not consider all relevant items but focuses on the average rank of the first relevant item. It is particularly useful when the topmost recommendations hold significant value.

:p What does MRR measure, and why might it be preferred in certain scenarios?
??x
MRR measures the mean reciprocal rank of the first relevant item, providing a simple yet interpretable metric for ranking performance. It is preferable in scenarios where the top recommendations are crucial and must be ranked high.

```java
public class Example {
    public double calculateMRR(double[] relevances) {
        // Implementation for calculating MRR
    }
}
```
x??

---

#### Correlation Coefficients

Background context: Correlation coefficients like Pearson’s or Spearman’s can measure the similarity between two rankings, such as predicted and ground-truth rankings. However, they do not provide the same information as mAP, MRR, or NDCG.

:p What is the role of correlation coefficients in evaluating ranking quality?
??x
Correlation coefficients are used to measure the degree of linear association between two continuous variables, which can indicate the overall similarity between ordered lists but do not directly evaluate precision and recall metrics like mAP, MRR, or NDCG.

```java
public class Example {
    public double calculatePearsonsCorrelation(double[] predictions, double[] groundTruth) {
        // Implementation for calculating Pearson's correlation coefficient
    }
}
```
x??

---

#### Offline Evaluation Methodologies

Background context: Offline evaluation methodologies are crucial for assessing the performance of recommendation algorithms before deploying them in real-world scenarios. These methods include metrics like mAP, MRR, NDCG, and correlation coefficients.

:p What is the purpose of offline evaluations in the context of recommendation systems?
??x
Offline evaluations help in understanding how well a recommendation algorithm performs using historical data before deployment. They allow for the assessment of various performance aspects such as precision, recall, relevance, and overall ranking quality through metrics like mAP, MRR, NDCG, and correlation coefficients.

```java
public class Example {
    public void performOfflineEvaluation(double[] predictions, double[] groundTruth) {
        // Implementation for performing offline evaluations using multiple metrics
    }
}
```
x??

---

