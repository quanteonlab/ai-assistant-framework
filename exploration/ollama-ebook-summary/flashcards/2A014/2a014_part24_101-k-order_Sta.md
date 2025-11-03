# Flashcards: 2A014 (Part 24)

**Starting Chapter:** 101-k-order Statistic

---

#### Classification for Ranking

Background context: One way to pose the ranking problem is as a multilabel task. Every item appearing in the training set that is associated with the user is labeled positively, while those outside are labeled negatively.

If we use a linear model, if \( X \) is the item vector and \( Y \) is the output, we learn \( W \), where sigmoid \( WX = 1 \) for items in the positive set; otherwise, sigmoid \( WX = 0 \).

This corresponds to the binary cross-entropy loss in Optax.

:p What is the key concept of using a multilabel approach for ranking?
??x
The key concept is that every item associated with the user in the training set is labeled positively, while items outside this set are labeled negatively. This setup allows us to train a model that predicts relevance but doesn't explicitly consider the relative ordering of items.
x??

#### Regression for Ranking

Background context: Another approach to ranking is through regression where we aim to predict ranks directly. For example, we can pose the problem as regressing towards NDCG given a query.

In practice, this involves conditioning the set of items against a query and using features of both the query and items. The goal is to learn a model that predicts relevance scores that reflect the actual ranking preferences.

If we use a linear model, if \( X \) is the item vector and \( Y \) is the output, then we learn \( W \), where \( W Xi = NDCGi \) and \( NDCGi \) is the NDCG for item \( i \).

Regression can be learned using L2 loss in Optax.

:p What does regression for ranking aim to achieve?
??x
The key concept is that regression for ranking aims to directly predict the relevance scores of items, reflecting their actual rankings. This approach helps in optimizing personalization metrics but doesn't explicitly consider the relative ordering of items.
x??

#### WARP (Weighted Approximate Rank Pairwise)

Background context: WARP (Weighted Approximate Rank Pairwise) is a loss function that approximates ranking by breaking it into pairwise comparisons.

If we have positive and negative item vectors \( X_{pos} \) and \( X_{neg} \), the model learns \( W \) such that \( WX_{pos} - WX_{neg} > 1 \).

The loss for this is hinge loss where the predictor output is \( WX_{pos} - WX_{neg} \) and the target is 1.

To adjust for unobserved items, we count how many times we had to sample from the negative set before finding a violating pair. This helps in assigning appropriate weights to different pairs.

:p What does WARP loss do?
??x
WARP loss optimizes precision@k by comparing higher-ranked and lower-ranked items. It applies hinge loss to pairs where the score of a higher-ranked item should be greater than that of a lower-ranked item by at least 1. The weight for each pair is adjusted based on how difficult it was to find a violating negative.
x??

#### k-order Statistic Loss

Background context: k-order statistic loss generalizes WARP and hinge loss by considering multiple positive items during the gradient step, providing a spectrum of loss functions.

Instead of just one positive item, several positive samples are considered. This allows for optimizing different aspects of ranking such as mean maximum rank or AUC loss.

The NumPy function `np.random.choice` can be used to sample from a distribution P that skews towards higher or lower ranks in the positive set.

:p How does k-order statistic loss differ from WARP and hinge loss?
??x
k-order statistic loss differs by considering multiple positive items during the gradient step, allowing for optimization of different ranking metrics like mean maximum rank. Unlike WARP which focuses on a single positive item, this approach provides more flexibility in choosing the positives based on the distribution P.
x??

#### Stochastic Losses and GPUs

Background context: Stochastic losses like WARP were designed for CPUs where sampling was cheap. However, with modern GPUs, branching decisions used in these stochastic methods can be less efficient.

On GPUs, threads need to run the same code over different data in parallel, making early exits from branches less effective.

:p Why are stochastic losses less efficient on modern GPUs?
??x
Stochastic losses like WARP become less efficient on modern GPUs because GPU cores process many data points in parallel. Early exits from branching decisions mean that both sides of a branch must be run, reducing the computational savings achieved through these early exits.
x??

---

#### BM25 Overview
BM25 is an algorithm used for ranking documents based on their relevance to a given query. It combines term frequency (TF) and inverse document frequency (IDF) to calculate the score of each document, taking into account the length normalization of the documents as well.
Relevant formula:
\[ \text{scoreD,Q} = \sum_{i=1}^n \frac{\text{IDF}_{qi} \cdot f_{qi,D}}{(k_1 + 1) \cdot (f_{qi,D} + k_1 \cdot (1 - b + b \cdot D / \text{avgdl}))} \]
where:
- \(D\) is the document.
- \(Q = q_1, q_2, \ldots, q_n\) is the query with terms \(q_i\).
- \(f_{qi,D}\) is the frequency of term \(q_i\) in document \(D\).
- \(D\) is the length of document \(D\).
- \(\text{avgdl}\) is the average document length.
- \(k_1\) and \(b\) are hyperparameters.
- \(\text{IDF}_{qi} = \log\left( \frac{N - n_{qi} + 0.5}{n_{qi} + 0.5} \right)\), where \(N\) is the total number of documents in the collection, and \(n_{qi}\) is the number of documents containing term \(q_i\).

:p What does BM25 stand for, and what is its primary purpose?
??x
BM25 stands for Best Matching 25. Its primary purpose is to rank documents based on their relevance to a given query by considering factors like term frequency (TF) and inverse document frequency (IDF), while also normalizing the length of the documents.
x??

---

#### BM25 Formula Breakdown
The formula for calculating the score in BM25 considers both the term frequency (how often a term appears in a document) and the inverse document frequency (how much unique information a term provides, measured by its rarity across all documents). It also includes length normalization to prevent longer documents from dominating shorter ones.
Relevant formula:
\[ \text{scoreD,Q} = \sum_{i=1}^n \frac{\text{IDF}_{qi} \cdot f_{qi,D}}{(k_1 + 1) \cdot (f_{qi,D} + k_1 \cdot (1 - b + b \cdot D / \text{avgdl}))} \]
where:
- \(D\) is the document.
- \(Q = q_1, q_2, \ldots, q_n\) is the query with terms \(q_i\).
- \(f_{qi,D}\) is the frequency of term \(q_i\) in document \(D\).
- \(D\) is the length of document \(D\).
- \(\text{avgdl}\) is the average document length.
- \(k_1\) and \(b\) are hyperparameters.

:p What is the BM25 formula, and what do its components represent?
??x
The BM25 formula is:
\[ \text{scoreD,Q} = \sum_{i=1}^n \frac{\text{IDF}_{qi} \cdot f_{qi,D}}{(k_1 + 1) \cdot (f_{qi,D} + k_1 \cdot (1 - b + b \cdot D / \text{avgdl}))} \]
where:
- \(D\) is the document.
- \(Q = q_1, q_2, \ldots, q_n\) is the query with terms \(q_i\).
- \(f_{qi,D}\) is the frequency of term \(q_i\) in document \(D\).
- \(D\) is the length of document \(D\).
- \(\text{avgdl}\) is the average document length.
- \(k_1\) and \(b\) are hyperparameters.

This formula calculates a score for each document based on the query terms appearing in it, taking into account term frequency (TF), inverse document frequency (IDF), and document length normalization. The parameters \(k_1\) and \(b\) can be tuned to fit specific characteristics of the document set.
x??

---

#### BM25 Hyperparameters
BM25 uses two hyperparameters: \(k_1\), a positive tuning parameter that calibrates the scaling of document term frequency, and \(b\), which determines the length normalization:
- \(k_1\) is used to scale the term weight by the document's term frequency.
- \(b = 0.75\) (default) is often used for full scaling of term weight by document length.

:p What are the hyperparameters in BM25, and what do they do?
??x
BM25 has two key hyperparameters:
1. \(k_1\): A positive tuning parameter that calibrates the scaling of document term frequency.
2. \(b\): Determines the length normalization; typically set to 0.75 for full scaling.

These parameters help in fine-tuning the model based on the specific characteristics of the document collection and query sets.
x??

---

#### BM25 Implementation Steps
The general steps to integrate BM25 into a larger information retrieval system involve:
1. Retrieving candidate documents using BM25.
2. Computing features for each document, including the BM25 score.
3. Training or evaluating an LTR (Learning to Rank) model using these feature vectors and their relevance judgments.
4. Ranking the documents based on the scores generated by the LTR model.

:p How can we integrate BM25 with a Learning to Rank (LTR) model?
??x
To integrate BM25 with a Learning to Rank (LTR) model, follow these steps:
1. Retrieve candidate documents using BM25.
2. Compute features for each document, including the BM25 score.
3. Train or evaluate an LTR model using these feature vectors and their relevance judgments.
4. Rank the documents based on the scores generated by the LTR model.

This combination allows you to first narrow down a large collection of potential candidate documents and then fine-tune the ranking with more complex features and interactions.
x??

---

#### BM25 Score in Text Document Retrieval
Background context explaining that BM25 is a powerful baseline for text document retrieval. It provides a simple yet effective way to rank documents based on their relevance to queries, considering factors like term frequency and inverse document frequency.

BM25 score can be expressed as:
\[ \text{score}(q,d) = k_1 (1 - b + b \cdot \frac{|d|}{\text{avgdl}}) \sum_{t \in q} \text{tf}_{t,d} \cdot \log \left( \frac{\text{N} - \text{n}_t + 0.5}{\text{n}_t + 0.5} \right) \]
where:
- \( k_1 \) and \( b \) are parameters,
- \( |d| \) is the document length,
- \( \text{avgdl} \) is the average document length,
- \( \text{n}_t \) is the number of documents containing term \( t \),
- \( \text{N} \) is the total number of documents.

:p What does BM25 score represent in text retrieval?
??x
BM25 score represents a measure used to rank documents based on their relevance to a given query. It considers factors such as term frequency, inverse document frequency (IDF), and the document's length relative to average document lengths.
x??

---

#### Latent Model Integration for Retrieval
Background context explaining how latent models can be integrated into retrieval systems. The approach involves using BM25 scores initially to get a set of "anchors" and then querying these anchors with a latent model.

:p How does the proposed method integrate latent models in text retrieval?
??x
The proposed method integrates latent models by first using BM25 scores to retrieve an initial set of documents ("anchors"). Then, each anchor is queried individually using latent models. Finally, an LTR (Learning To Rank) model aggregates and ranks the results from these searches.

```java
public class LatentModelIntegration {
    public List<Document> searchWithBM25(String query) {
        // Implementation to retrieve anchors using BM25
        return new ArrayList<>();  // Placeholder for actual implementation
    }

    public void trainLTR(List<Document> initialSet, List<Document> latentResults) {
        // Train LTR model on the union of both sets
    }
}
```
x??

---

#### Multimodal Retrieval Approach
Background context explaining how multimodal retrieval leverages multiple latent spaces. The approach starts with an initial BM25 search to get anchors and then uses these as queries in a latent model, aggregating results.

:p What is the key benefit of using this multimodal retrieval approach?
??x
The key benefit of using this multimodal retrieval approach is that it leverages multiple latent spaces by first using BM25 to retrieve initial "anchors" and then querying each anchor with a latent model. This ensures that the search process considers both textual content and other modalities, making the system more robust and versatile.

```java
public class MultimodalRetrieval {
    public List<Document> multimodalSearch(String query) {
        List<String> anchors = searchWithBM25(query);
        List<Document> latentResults = new ArrayList<>();
        for (String anchor : anchors) {
            latentResults.addAll(searchWithLatentModel(anchor));
        }
        return trainLTR(anchors, latentResults);
    }

    private List<Document> searchWithBM25(String query) {
        // Implementation using BM25
        return new ArrayList<>();  // Placeholder for actual implementation
    }

    private List<Document> searchWithLatentModel(String anchor) {
        // Implementation to query the latent model
        return new ArrayList<>();  // Placeholder for actual implementation
    }

    private List<Document> trainLTR(List<String> initialSet, List<Document> latentResults) {
        // Train LTR model on the union of both sets and return ranked documents
        return new ArrayList<>();  // Placeholder for actual implementation
    }
}
```
x??

---

#### Query Out-of-Distribution Considerations
Background context explaining how queries can often be out of distribution from the document space, especially in encoder-based latent models. This highlights the importance of leveraging multiple modalities to ensure robustness.

:p How do encoder-based latent models handle queries that are out of distribution?
??x
Encoder-based latent models can struggle with queries that are significantly different from the training data (out of distribution). To address this, multimodal retrieval approaches leverage both textual content and other modalities. For example, a query like "Who’s the leader of Mozambique?" might not align well with article titles or sentences directly. By using BM25 to find relevant anchors and then querying latent models on these, the system can better handle such out-of-distribution queries.

```java
public class OutOfDistributionHandling {
    public Document retrieveDocument(String query) {
        List<String> anchors = searchWithBM25(query);
        for (String anchor : anchors) {
            Document doc = searchWithLatentModel(anchor);
            if (doc != null && isRelevant(doc, query)) {
                return doc;
            }
        }
        return null;  // Fallback if no relevant document found
    }

    private List<String> searchWithBM25(String query) {
        // Implementation using BM25
        return new ArrayList<>();  // Placeholder for actual implementation
    }

    private Document searchWithLatentModel(String anchor) {
        // Implementation to query the latent model
        return null;  // Placeholder for actual implementation
    }

    private boolean isRelevant(Document doc, String query) {
        // Logic to determine if document is relevant to query
        return false;  // Placeholder for actual implementation
    }
}
```
x??

---

#### Summary of Ranking Concepts
Background context summarizing the learning-to-rank concepts covered so far, including traditional methods like BM25 and more advanced techniques like WARP and WSABIE. The focus is on understanding the importance of ranking in recommendation systems.

:p What are the key takeaways from the ranking concepts discussed?
??x
Key takeaways include:
- **Learning to Rank (LTR)**: A fundamental concept where models learn to rank items based on their relevance.
- **BM25**: A traditional and effective baseline for text retrieval, considering term frequency, document length, and IDF.
- **WARP and WSABIE**: Advanced methods that improve upon BM25 by using more careful probabilistic sampling.
- **K-order Statistic**: Utilizing k-best samples to better approximate the ranking objective.

These concepts are essential in building robust recommendation systems where ordering items based on relevance is critical.

```java
public class Summary {
    public void summarizeRankingConcepts() {
        System.out.println("BM25 provides a strong baseline for text retrieval, "
                + "WARP and WSABIE offer improvements by using better sampling techniques. "
                + "LTR is crucial in ranking items based on their relevance.");
    }
}
```
x??

---

