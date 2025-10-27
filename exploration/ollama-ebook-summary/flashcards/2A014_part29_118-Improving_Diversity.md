# Flashcards: 2A014 (Part 29)

**Starting Chapter:** 118-Improving Diversity

---

#### Intra-list Diversity

In intra-list diversity, the goal is to ensure a variety of types of items within a single recommendation list. This approach aims to minimize the similarity between recommended items to reduce overspecialization and encourage exploration.

High intra-list diversity can increase user exposure to many items they may like; however, this reduces recall for specific interests since each interest is represented with fewer but more diverse items.

:p What is intra-list diversity in recommendations?
??x
Intra-list diversity aims to include a variety of different types of items within one recommendation list. By minimizing the similarity between recommended items, it promotes exploration over overspecialization and ensures that users are exposed to a broader range of content, even if they may not be as deeply relevant for each specific interest.

```java
public class RecommendationDiversity {
    public void enhanceIntraList(List<RecommendationItem> recommendations) {
        // Logic to ensure diversity in the recommendation list
        // Example: Calculate similarity between items and adjust based on a threshold
    }
}
```
x??

---

#### Serendipitous Recommendations

Serendipitous recommendations are designed to be both surprising and interesting to users. These can include items that the user might not have discovered independently or are generally less popular in the system.

To introduce serendipity, non-obvious or unexpected choices can be injected into the recommendation process, even if they have a relatively lower affinity score with the user. The goal is to improve overall serendipity while maintaining high affinity relative to other items of similar popularity.

:p What are serendipitous recommendations?
??x
Serendipitous recommendations aim to provide users with items that are both surprising and interesting, often including content they might not have discovered on their own or are less popular in the system. By incorporating non-obvious choices into the recommendation process, even if these options have a lower affinity score, it enhances overall serendipity while maintaining high affinity relative to other similar items.

```java
public class SerendipityInjector {
    public void injectSerendipity(List<RecommendationItem> recommendations) {
        // Logic to introduce unexpected choices into the recommendation list
        // Example: Randomly select less popular but highly rated items for inclusion
    }
}
```
x??

---

#### Reranking Strategy

Reranking is a post-processing step that reorders an initially retrieved recommendation list to enhance diversity. It considers both relevance scores and dissimilarity among items in the recommendation list.

By applying reranking, you can improve diversity metrics while potentially sacrificing performance on recall or NDCG. This strategy operationalizes any external loss function for diversity, making it straightforward to implement.

:p What is reranking in recommendations?
??x
Reranking is a post-processing step that reorders an initially retrieved recommendation list to enhance diversity. It takes into account both relevance scores and the dissimilarity among items in the recommendation list. By applying reranking, you can improve diversity metrics while potentially sacrificing performance on recall or NDCG.

```java
public class Reranker {
    public List<RecommendationItem> reRank(List<RecommendationItem> initialList) {
        // Logic to reorder the initial list based on diversity criteria
        // Example: Sort items by a combination of relevance and dissimilarity
        return sortedList;
    }
}
```
x??

---

#### Explore-Exploit Trade-off

In explore-exploit trade-offs, the recommendation system balances between exploiting known user preferences (choosing high-affinity options) and exploring less certain but potentially higher-reward choices. This can be implemented by using affinity as a reward estimate and propensity as an exploitation measure.

:p How does the explore-exploit trade-off work in recommendations?
??x
The explore-exploit trade-off in recommendation systems balances between exploiting known user preferences (choosing high-affinity options) and exploring less certain but potentially higher-reward choices. This can be implemented by using affinity as a reward estimate to exploit items the model is confident will be liked, and propensity as an exploitation measure to choose more uncertain or diverse options.

```java
public class ExploreExploit {
    public RecommendationItem recommendNext(User user) {
        double propensity = calculatePropensity(user);
        if (Math.random() < propensity) {
            // Exploitation: Recommend high-affinity item
            return highestAffinityItem(user);
        } else {
            // Exploration: Recommend less obvious choice
            return randomNonObviousItem();
        }
    }

    private double calculatePropensity(User user) {
        // Logic to determine the probability of exploration based on user behavior
        return 0.2; // Example value
    }

    private RecommendationItem highestAffinityItem(User user) {
        // Logic to select high-affinity item for user
        return topRatedItem(user);
    }

    private RecommendationItem randomNonObviousItem() {
        // Logic to randomly select a less obvious recommendation
        return getRandomItem();
    }
}
```
x??

---

#### Multimodal Recommendations

Multimodal recommendations integrate various ranking measures from different domains to suggest items from outside the user's "mode," thus broadening the range of recommendations. This approach uses multiple query vectors for each request, forcing self-similarity among the retrieved list and promoting diversity.

:p What are multimodal recommendations?
??x
Multimodal recommendations integrate various ranking measures from different domains to suggest items from outside the user’s “mode,” thereby broadening the range of recommendations. By using multiple query vectors for each request, this approach promotes diversity and ensures that the recommendation system suggests a wider variety of content.

```java
public class MultimodalRecommender {
    public List<RecommendationItem> recommendMultimodal(User user) {
        // Logic to use multiple query vectors for each user
        List<VectorQuery> queries = generateQueries(user);
        return integrateRankings(queries);
    }

    private List<VectorQuery> generateQueries(User user) {
        // Generate different query vectors based on the user's context
        return Arrays.asList(query1, query2, ...);
    }

    private List<RecommendationItem> integrateRankings(List<VectorQuery> queries) {
        // Integrate rankings from multiple queries to provide diverse recommendations
        List<RecommendationItem> integratedList = new ArrayList<>();
        for (VectorQuery query : queries) {
            List<RecommendationItem> rankedItems = rank(query);
            integratedList.addAll(rankedItems);
        }
        return integrateAndFilter(integratedList);
    }

    private List<RecommendationItem> rank(VectorQuery query) {
        // Logic to rank items based on the query vector
        return rankedItems;
    }

    private List<RecommendationItem> integrateAndFilter(List<RecommendationItem> integratedList) {
        // Integrate and filter the list of recommendations
        return filteredList;
    }
}
```
x??

---

#### Portfolio Optimization for Recommendation Systems
Portfolio optimization, inspired by finance, is a technique to enhance diversity in recommendation systems. It balances relevance (risk) and diversity (return). The core idea involves creating a "portfolio" of items that optimally balance these two parameters.

Formulate item representations where the distance between them serves as a measure of similarity.
Calculate pairwise distances between all retrieved items using a chosen metric.
Evaluate affinity scores to better estimate returns.
Solve an optimization problem to find weights for each item. The objective function is:
\[ \text{Maximize } w^T r - \lambda w^T C w \]
Where \( w \) represents the weights, \( r \) the relevance score vector, and \( C \) the covariance matrix capturing diversity. \( \lambda \) balances these two metrics.

:p How does portfolio optimization work in recommendation systems?
??x
Portfolio optimization works by balancing relevance (relevance scores) and diversity (captured by a covariance matrix). The goal is to find weights for each item that maximize overall value while considering both factors. This ensures recommendations are not only relevant but also diverse.
```java
// Example pseudo-code for solving the optimization problem
public class PortfolioOptimizer {
    public double[] optimizeWeights(double[] relevanceScores, Matrix covarianceMatrix, double lambda) {
        // Implement optimization logic here
        return weights;
    }
}
```
x??

---

#### Multiobjective Functions in Recommendation Systems
Multiobjective functions are used to enhance diversity by incorporating multiple ranking terms. For instance, balancing personalization with image similarity can yield more diverse and relevant recommendations.

In the fashion recommender example, two latent spaces were utilized: one for personalized clothes and another for images of clothing.
A simple multiobjective function is:
\[ s_i = \alpha \times (1 - d_i) + 1 - \alpha \times a_i \]
Where \( \alpha \) represents the weighting between image similarity and personalization, \( d_i \) is the image distance, and \( a_i \) is the personalization score.

:p How does multiobjective ranking help in recommendation systems?
??x
Multiobjective ranking helps by balancing multiple criteria to ensure recommendations are both relevant and diverse. By using a function like \( s_i = \alpha \times (1 - d_i) + 1 - \alpha \times a_i \), the system can prioritize items that satisfy multiple conditions, such as image similarity and personalization.
```java
// Pseudo-code for applying multiobjective ranking
public double rankRecommendations(List<Item> items, double alpha, List<Double> distances, List<Double> personalizations) {
    List<RankedItem> rankedItems = new ArrayList<>();
    for (int i = 0; i < items.size(); i++) {
        RankedItem ri = new RankedItem(items.get(i), calculateScore(distances.get(i), personalizations.get(i), alpha));
        rankedItems.add(ri);
    }
    // Sort and return top-k
    Collections.sort(rankedItems, (o1, o2) -> Double.compare(o2.score, o1.score));
    return rankedItems;
}
```
x??

---

#### Predicate Pushdown for Recommendation Systems
Predicate pushdown is an optimization technique used in databases to filter data early in the retrieval process. This reduces the amount of data processed later in query execution.

In recommendation systems, predicate pushdown can be applied by filtering items based on specific features (e.g., color) before full scoring.
For example, if you want a diverse set of at least three colors, perform top- \( k \) searches for each color and then rank the union of these sets.

:p How does predicate pushdown help in recommendation systems?
??x
Predicate pushdown helps by reducing the amount of data processed during retrieval. By filtering items based on specific features early, you can significantly decrease the number of full-score evaluations needed, improving efficiency without compromising diversity.
```java
// Pseudo-code for predicate pushdown
public List<Item> applyPredicatePushdown(List<Item> items, Set<String> requiredColors) {
    Map<String, List<Item>> colorGroups = new HashMap<>();
    // Group by color and select top-k in each group
    for (Item item : items) {
        if (!colorGroups.containsKey(item.color)) {
            colorGroups.put(item.color, new ArrayList<>());
        }
        colorGroups.get(item.color).add(item);
    }
    List<Item> filteredItems = new ArrayList<>();
    for (String color : requiredColors) {
        // Select top-k from each group
        Collections.sort(colorGroups.get(color), (o1, o2) -> Double.compare(o2.score, o1.score));
        int k = 3; // Example value for k
        List<Item> topK = new ArrayList<>(colorGroups.get(color).subList(0, Math.min(k, colorGroups.get(color).size())));
        filteredItems.addAll(topK);
    }
    return filteredItems;
}
```
x??

---

#### Fairness in Recommendation Systems
Fairness in recommendation systems is a critical aspect of ensuring unbiased and equitable outcomes. Techniques like nudging can be used to emphasize certain behaviors or buying patterns.

Filter bubbles are a common downside where users get similar recommendations, leading to limited exposure. Mitigation strategies include awareness and diverse content curation.
High-risk applications require careful management and robust safeguards.

:p How does fairness impact recommendation systems?
??x
Fairness in recommendation systems impacts the system's ability to provide unbiased and equitable outcomes. Techniques like nudging can ensure that certain behaviors or buying patterns are promoted, reducing bias. Filter bubbles must be avoided to ensure users receive a diverse range of recommendations.
```java
// Example pseudo-code for fairness mitigation
public void mitigateFilterBubbles(List<Item> recommendations) {
    // Implement logic to diversify recommendations
    Collections.shuffle(recommendations); // Shuffle items to break filter bubble patterns
}
```
x??

--- 
Note: These flashcards cover the key concepts in a detailed and educational manner, providing context and explanations rather than purely memorization. Code examples are provided where relevant to illustrate the logic behind each concept.

#### Sharding Strategy
Sharding is a method to divide and conquer by distributing data across multiple machines. It can reduce runtime complexity from O(N * M) to O(N * M / k), where \(k\) represents the number of machines.
:p How does sharding help in speeding up recommendation systems?
??x
Sharding helps by dividing the workload among multiple machines, allowing parallel processing and thus reducing the overall computation time. Each machine handles a portion of the data, and when recommendations are needed for a user, each machine computes its part independently before results are combined.
```python
# Pseudocode example to demonstrate sharding
def assign_to_machine(unique_id: int, k: int) -> int:
    """Assigns an item uniquely to one of k machines."""
    return unique_id % k

# Example usage
k = 4  # Number of machines
unique_ids = [10, 20, 30, 40, 50]
machines = {}
for id in unique_ids:
    machine_id = assign_to_machine(id, k)
    if machine_id not in machines:
        machines[machine_id] = []
    machines[machine_id].append(id)

print(machines)
```
x??

---

#### Locality Sensitive Hashing (LSH) Concept
Locality Sensitive Hashing is a technique to convert vector representations into token-based hashes, enabling faster similarity searches. The key property of LSH is that similar vectors should have the same hash codes.
:p How does LSH help in speeding up the search for similar items?
??x
LSH helps by converting vector representations into token-based hashes, which can be more efficiently compared using integer arithmetic operations like XOR and bit counting. This allows for faster similarity searches as regular database search engines can leverage hash matching to find nearby vectors.
```python
# Pseudocode example of LSH computation
def compute_wta_hash(x):
    """Computes a Winner Take All (WTA) hash code."""
    key = jax.random.PRNGKey(1337)
    permuted = jax.random.permutation(key, x)
    hash1 = permuted[0] > permuted[1]
    hash2 = permuted[1] > permuted[2]
    return (hash1, hash2)

# Example usage
x1 = jnp.array([1, 2, 3])
x2 = jnp.array([1, 2.5, 3])
x3 = jnp.array([3, 2, 1])

x1_hash = compute_wta_hash(x1)
x2_hash = compute_wta_hash(x2)
x3_hash = compute_wta_hash(x3)

print(x1_hash)  # Output: (False, True)
print(x2_hash)  # Output: (False, True)
print(x3_hash)  # Output: (True, False)
```
x??

---

#### Hamming Distance in LSH
Hamming distance is used to compute the similarity between hash codes generated by LSH. It calculates the number of differing bits between two binary strings.
:p How do you calculate the distance using Hamming distance?
??x
The Hamming distance can be calculated as the XOR of two hash codes followed by bit counting, which gives the number of positions at which the corresponding bits are different.
```python
# Pseudocode example to compute Hamming distance
def hamming_distance(hash1: tuple, hash2: tuple) -> int:
    """Computes the Hamming distance between two hash codes."""
    xor_result = (hash1[0] ^ hash2[0]), (hash1[1] ^ hash2[1])
    return sum(x for x in xor_result)

# Example usage
hash1 = (False, True)
hash2 = (False, True)
distance = hamming_distance(hash1, hash2)
print(distance)  # Output: 0

hash3 = (True, False)
distance = hamming_distance(hash1, hash3)
print(distance)  # Output: 2
```
x??

---

