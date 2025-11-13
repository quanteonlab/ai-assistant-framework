# Flashcards: 2A014 (Part 4)

**Starting Chapter:** 21-Summary

---

#### Business Insight and What People Like

Background context: The passage explains how businesses can leverage recommendation systems to provide business insights beyond just making recommendations. It discusses the importance of understanding trends, KPIs, and primary drivers of engagement, using Netflix's Squid Game as an example.

:p How does a company use recommendation systems for business insights?
??x
A company uses recommendation systems to understand weekly trends and key performance indicators (KPIs) such as weekly active users and new sign-ups. For instance, Netflix used its recommendation system to identify that the show "Squid Game" became their most popular series of all time, breaking multiple records in a short period. By analyzing this data, executives can derive insights such as investing more in shows from South Korea or subtitled shows with high drama.

Code example (pseudocode):
```java
// Pseudocode to analyze weekly trends and KPIs
if (viewershipOfSquidGame > previousWeekViewership) {
    System.out.println("Invest more in similar content");
} else if (viewershipOfSquidGame < expectedValue) {
    System.out.println("Analyze why the growth metrics are lagging");
}
```
x??

---

#### Top Sellers of the Week

Background context: The passage provides an example of using a recommender system to create a "Top Sellers of the Week" carousel. This is done by applying a popular recommendation algorithm (e.g., get_most_popular_recs) to a specific collector that only considers items from the last week.

:p What does the "Top Sellers of the Week" carousel represent?
??x
The "Top Sellers of the Week" carousel represents a recommender system applied specifically to the most popular items in the previous week. It is an example of combining recommendation with business insight, as it helps understand trends and KPIs like weekly active users.

Code example (pseudocode):
```java
// Pseudocode for Top Sellers of the Week
public List<Item> getTopSellersOfWeek() {
    return recommender.getMostPopularItemsCollector(lastWeek);
}
```
x??

---

#### Incremental Gains

Background context: The passage introduces the concept of "incremental gains" in growth marketing and analytics. It explains how businesses can measure additional growth beyond their expected effort.

:p What is incremental gain, and why is it important?
??x
Incremental gain refers to an increase in business results above what was initially expected from a given effort. In marketing, for example, if a business usually adds one user per $100 spent on marketing but gets more users with the same budget due to positive press or other factors, this excess is considered incremental gain.

Code example (pseudocode):
```java
// Pseudocode to calculate incremental gains
public int calculateIncrementalGain(int initialUsersPerBudget, int actualUsersPerBudget) {
    return Math.max(0, actualUsersPerBudget - initialUsersPerBudget);
}
```
x??

---

#### Diversifying Recommendations

Background context: The passage mentions that diversifying recommendations can increase the overall ability to match users with items and grow future opportunities.

:p Why is diversity important in recommendation systems?
??x
Diversity in recommendations ensures a broader range of content, which keeps a broad base of users highly engaged. This increases the future opportunity for growth as different user groups are more likely to find something relevant or interesting among a diverse set of options.

Code example (pseudocode):
```java
// Pseudocode for diversifying recommendations
public List<Item> getDiverseRecommendations(User user, int numItems) {
    List<Item> popularItems = recommender.getMostPopularItems(numItems);
    List<Item> relatedItems = recommender.getItemsRelatedToUserInterests(user, numItems);
    return Stream.concat(popularItems.stream(), relatedItems.stream())
                 .distinct()
                 .limit(numItems)
                 .collect(Collectors.toList());
}
```
x??

---

#### Zipf’s Laws in RecSys and the Matthew Effect
Background context: In many recommendation systems, especially those dealing with large datasets like MovieLens, data often follows a Zipfian distribution. This means that items (e.g., movies) are ranked such that their frequency decreases exponentially. Additionally, popular items receive much higher engagement rates than less popular ones, a phenomenon known as the Matthew effect or popularity bias.

Mathematically, this can be expressed using Zipf’s law:
$$f_k,M = \frac{1}{k} \sum_{n=1}^M \frac{1}{n}$$

Where $k $ is the rank of an item and$M$ is the total number of items.

The Matthew effect can be observed in how popular items attract more attention, widening the gap between them and less popular ones. For example, in MovieLens data, very popular movies have significantly higher click counts compared to average movies.

:p What does Zipf’s law describe in recommendation systems?
??x
Zipf’s law describes a distribution where the frequency of occurrence decreases exponentially with rank. In the context of recommendation systems, it means that fewer items will receive much more engagement than many other less popular items.
x??

---
#### The Matthew Effect or Popularity Bias
Background context: The Matthew effect states that popular items tend to attract more attention and widen the gap between them and others in terms of popularity.

:p What is the Matthew effect (popularity bias) in recommendation systems?
??x
The Matthew effect, or popularity bias, means that the most popular items continue to receive disproportionate amounts of attention, thus increasing their visibility even further compared to less popular items.
x??

---
#### Impact on User-Based Collaborative Filtering
Background context: When building user-based collaborative filtering models (UBCF), the distributional properties of data and the Matthew effect can significantly impact recommendation outcomes. These effects can lead to biased recommendations favoring highly popular users or items.

Formula for joint probability of an item appearing in two user's ratings:
$$P_i = \frac{f_{i,M}}{\sum_{j=1}^{M} f_{j,M}} = \frac{1/i}{\sum_{j=1}^{M} 1/j}$$:p How does the Matthew effect impact UBCF models?
??x
The Matthew effect impacts UBCF by causing popular items to be disproportionately recommended because they have higher joint probability scores. This is due to the formula for similarity, where the probability of shared items between two users decreases rapidly as the popularity rank increases.
x??

---
#### Similarity Calculation in UBCF Models
Background context: In user-based collaborative filtering (UBCF), the similarity between users is often calculated based on the number of jointly rated items. The Matthew effect influences this calculation, leading to lower similarity scores for less popular items.

Formula for joint probability of shared items:
$$

P_{i2} = \frac{1/i^2}{\sum_{j=1}^{M} 1/j^2}$$:p How does the Matthew effect affect the similarity score in UBCF?
??x
The Matthew effect affects the similarity score by reducing it for less popular items. Since the joint probability of shared items decreases with the square of their popularity rank, less popular items are rated lower in terms of similarity, leading to biased recommendations.
x??

---
#### Exemplification with Last.fm Dataset
Background context: The paper "Quantitative Analysis of Matthew Effect and Sparsity Problem of Recommender Systems" by Hao Wang et al. demonstrates how the Matthew effect impacts the similarity matrix using the Last.fm dataset. This dataset tracks music listening behavior, showing persistent effects even in the similarity scores.

:p How does the Matthew effect manifest in the Last.fm dataset?
??x
The Matthew effect manifests in the Last.fm dataset through persistent bias in similarity scores, where popular users or items maintain higher engagement and thus have a greater influence on recommendations. This is evident when analyzing the matrix of user-item interactions.
x??

---
#### Combinatorial Formulas for UBCF
Background context: The combinatorial formulas indicate that the Zipfian distribution significantly impacts the output scores in collaborative filtering models, particularly UBCF.

Formula for similarity score calculation:
$$\sum_{i=1}^{M} P_i^2 / \| \mathcal{I}_A \cup \mathcal{I}_B \|$$:p What are combinatorial formulas used for in UBCF?
??x
Combinatorial formulas are used to quantify the impact of the Zipfian distribution on collaborative filtering models. They help in understanding how the popularity rank affects similarity scores, leading to more accurate but potentially biased recommendations.
x??

---

#### Matthew Effect

Background context: The Matthew effect refers to a phenomenon where popular items become even more popular, while less popular items are often ignored. This is observed through sparse data, with some extremely popular items dominating the dataset.

:p What does the Matthew effect describe in recommendation systems?
??x
The Matthew effect describes how certain popular items receive disproportionately high ratings or interactions compared to less popular items, leading to a skewed distribution of data.
x??

---

#### Sparsity

Background context: As ratings skew towards the most popular items, less popular items are represented sparsely in the dataset. This leads to sparse vectors and matrices, which are challenging for recommendation algorithms.

:p What is sparsity in the context of recommendation systems?
??x
Sparsity in recommendation systems refers to the situation where data is predominantly missing or zero-valued, especially for less popular items. It creates challenges because many user-item interactions are unknown.
x??

---

#### User Similarity Counts

Background context: The user similarity counts help understand how users with similar preferences interact. This concept is critical as it affects recommendation algorithms by highlighting the influence of highly rated or popular items.

:p How does the formula for user similarity count work?
??x
The formula for calculating the expected number of other users who click the ith most popular item, given by $P_i = \frac{f_{i,M}}{\sum_{j=1}^{M} f_{j,M}} = \frac{1/i}{\sum_{j=1}^{M} 1/j}$, helps determine the popularity of an item and its influence on other users. The total number of users who will share a rating with X can be calculated as $\sum_{i=1}^{M} (M-1) * P_i$.

This formula accounts for the decreasing probability that less popular items are rated by many users, thus influencing user similarity counts.
x??

---

#### Visual Representations

Background context: Figures 3-2 and 3-3 illustrate the Matthew effect and sparsity in the Last.fm dataset. These visualizations show how some items dominate while others are rarely interacted with.

:p What do Figure 3-2 and 3-3 reveal about recommendation systems?
??x
Figure 3-2 reveals that a few extremely popular items overshadow many more less popular ones, leading to a highly skewed distribution of interactions. Figure 3-3 shows the sparsity in user-item interactions, with fewer ratings for less popular items.

These visualizations highlight the need for addressing both the Matthew effect and sparsity in recommendation systems.
x??

---

#### Impact on CF Algorithms

Background context: The sparsity introduced by the Matthew effect affects collaborative filtering (CF) algorithms. Users of different ranks are used to compute similarity, but due to sparsity, highly popular items dominate user-item interactions.

:p How does sparsity impact collaborative filtering algorithms?
??x
Sparsity impacts collaborative filtering algorithms significantly because less popular items receive fewer ratings, leading to sparse user-item matrices. This means that many users share similar preferences for the same set of highly rated items, thus overshadowing less popular ones in similarity calculations.

To address this, downstream sampling methods and diversity-aware loss functions can be employed.
x??

---

