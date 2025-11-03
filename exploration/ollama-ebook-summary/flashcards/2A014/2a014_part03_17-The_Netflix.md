# Flashcards: 2A014 (Part 3)

**Starting Chapter:** 17-The Netflix Challenge

---

#### Matrix Representation and Data Visualization
Background context: The provided data is presented as a matrix, where indices represent user-item interactions, and values represent ratings. This representation allows for an efficient way to capture and analyze user preferences.

:p What does the given matrix represent?
??x
The matrix represents a set of user-item interactions, where each index (i,j) corresponds to a specific user-item pair, and the value at that position indicates the rating given by user i to item j.
For example:
```python
indices = [(0,0),(0,1),(0,2),(0,3),
           (1,0),(1,1),(1,2),(1,3),
           (2,0),(2,1),(2,2),(2,3),
           (3,0),(3,1),(3,2),
           (4,0)]
values = [5,4,4,1,
          2,3,3,4.5,
          3,2,3,4,
          4,4,5,
          3]
```
x??

---

#### Most Popular Cheese
Background context: The example given suggests an analysis of cheese preferences based on user ratings. It highlights the use of this matrix representation to derive insights about popular items and individual preferences.

:p What is the most popular cheese according to the provided data?
??x
Based on the ratings, Emmentaler seems to be a favorite since it has one of the highest values (5), but user E did not try it. The exact conclusion depends on the overall context and additional data.
x??

---

#### Predicting User Preferences Using Matrix Completion
Background context: The example poses questions that hint at the concept of matrix completion, where missing ratings are predicted based on known ones.

:p How does the problem of predicting "how much a user will like an item they haven't seen" relate to matrix completion?
??x
The problem relates to matrix completion by trying to fill in unknown elements (user-item pairs with no rating) using known data. This is akin to predicting missing values in a matrix based on its existing entries.
x??

---

#### User-User Versus Item-Item Collaborative Filtering
Background context: The text explains two collaborative filtering strategies—user-user and item-item—and their mathematical interpretations.

:p How does user-user collaborative filtering work?
??x
In user-user CF, the approach is to find a similar user (a user with similar tastes) and then recommend items that this similar user has liked but the target user hasn't seen yet.
x??

---

#### Item-Item Collaborative Filtering
Background context: Similar to the previous card, item-item collaborative filtering involves finding similar items based on users who have interacted with them.

:p How does item-item collaborative filtering work?
??x
In item-item CF, the approach is to find an item that a user has liked and then recommend other similar items that this user hasn't seen yet.
x??

---

#### Vector Similarity and Cosine Similarity
Background context: The text mentions vector similarity being computed by normalizing vectors and taking their cosine similarity.

:p How is vector similarity typically calculated?
??x
Vector similarity, particularly in the context of collaborative filtering, is often calculated using cosine similarity after normalizing the vectors. This involves measuring the cosine of the angle between two vectors to determine how similar they are.
```java
// Pseudocode for computing cosine similarity
public double computeCosineSimilarity(Vector v1, Vector v2) {
    double dotProduct = v1.dot(v2);
    double magnitudeV1 = Math.sqrt(v1.dot(v1));
    double magnitudeV2 = Math.sqrt(v2.dot(v2));
    return dotProduct / (magnitudeV1 * magnitudeV2);
}
```
x??

---

#### The Netflix Challenge
Background context: This section describes the history and details of the Netflix Prize competition, highlighting its importance in the recommendation system community.

:p What was the goal of the Netflix Prize competition?
??x
The goal of the Netflix Prize competition was to improve on Netflix's own CF algorithms by reducing the root mean square error (RMSE) of predictions for user-movie ratings. Teams were challenged to achieve a 10% improvement over Netflix’s internal performance metrics.
x??

---

#### Key Lessons from the Netflix Challenge
Background context: The text outlines several important lessons learned from participating in the Netflix Prize, including the importance of model selection, parameter tuning, and how business needs can change.

:p What is one key takeaway from the Netflix Prize competition?
??x
One key takeaway is that it's crucial to build a working usable model quickly and iterate while the model remains relevant to the business needs. This involves selecting an appropriate model, tuning parameters effectively, and continuously adapting to changing business circumstances.
x??

---

#### Hard vs Soft Ratings
Background context explaining the difference between hard and soft ratings. Hard ratings are explicit numerical values given by users, while soft ratings are implicit data indicating user behavior without direct feedback.

:p What is the main difference between hard and soft ratings?
??x
Hard ratings are explicit numerical values provided directly by users in response to a prompt, whereas soft ratings reflect user behavior that indirectly communicates their preferences or interest in an item.
x??

---

#### Implicit Ratings in Recommendation Systems
Background context on how implicit ratings can be used in recommendation systems. Discuss the importance of using data from behaviors like watching a movie without rating it.

:p How do implicit ratings differ from hard ratings in recommendation systems?
??x
Implicit ratings are based on user behavior that indirectly communicates preferences or interest, such as viewing content without explicitly rating it. Hard ratings are direct numerical feedback provided by users.
x??

---

#### Data Collection for Recommendation Systems
Explanation of how to collect data for recommendation systems through user interactions like page loads and clicks.

:p What types of data should be collected for building effective recommendation systems?
??x
For building effective recommendation systems, collect data from user interactions such as page loads, clicks, hover events, and other behaviors that indicate interest in items.
x??

---

#### Propensity Scores in A/B Testing
Explanation of propensity scores and how they relate to feature-stratified A/B testing.

:p What are propensity scores in the context of A/B testing?
??x
Propensity scores in A/B testing are the probabilities of an observational unit being assigned to a treatment group versus a control group. They help in adjusting for biases by matching units with similar probabilities.
x??

---

#### Logging User Interactions on Bookshop.org
Discussion on logging user interactions such as page loads, hover events, and clicks.

:p What types of interactions should be logged on an ecommerce site like Bookshop.org?
??x
On an ecommerce site like Bookshop.org, log interactions such as page loads (initial content displayed), hover events (mouse movements over items), and clicks. These interactions provide implicit feedback about user interest.
x??

---

#### Clicks vs Hover Events
Explanation of how clicks and hover events differ in indicating user intent.

:p How do clicks and hover events differ in terms of indicating user interest?
??x
Clicks are a strong indicator of high-level interest, as the user actively selects an item. Hover events provide lower-level engagement, indicating that the user is interested enough to interact more closely with the item but not necessarily select it.
x??

---

#### Importance of Logging All Page Loads
Explanation on why logging all page loads is crucial for understanding implicit ratings.

:p Why is logging all page loads important in recommendation systems?
??x
Logging all page loads is important because it helps capture the entire population of items a user has seen, which can provide context for implicit ratings. It allows the system to weigh interactions based on the number of options exposed.
x??

---

#### Assigning Default Ratings for Unrated Items
Discussion on assigning default ratings to items that have been observed but not rated.

:p How should unrated items be handled in a recommendation system?
??x
Unrated items should be handled by either excluding them from recommendations, using implicit data as separate terms, or assigning a default rating like "interesting enough to watch." This helps provide a balanced view of user interest.
x??

---

#### Multilevel Models for Recommendation Systems
Explanation on the use of multilevel models to predict both click and buy likelihood.

:p Why are multilevel models important in recommendation systems?
??x
Multilevel models are important because they can predict both click and buy likelihood, providing a more comprehensive understanding of user behavior. This is crucial for effective recommendations that consider various levels of interaction.
x??

---

#### Click Behavior and Its Significance

Background context: Understanding click behavior is crucial for many recommendation systems. Clicks often serve as a proxy for user intent, especially when purchasing something online. However, there's noise involved, but clicks remain a primary indicator of interest.

:p Why are clicks important in recommendation systems?
??x
Clicks are essential because they represent explicit user actions and can be used to train models effectively due to the high volume of data generated compared to other forms of interaction like ratings.
x??

---

#### Add-to-Bag as a Strong Indicator

Background context: Adding an item to a shopping cart is considered one of the strongest indicators of interest before completing a purchase. It marks a significant point in the user journey, often correlating closely with actual purchases.

:p Why is "add-to-bag" important for recommendation systems?
??x
Add-to-bag signifies a stronger level of intent towards purchasing than just clicking or viewing items. It's a more definitive action that typically precedes actual purchase, making it a better signal than simple clicks.
x??

---

#### Impression Data and Its Use

Background context: Impressions refer to the instances where an item is shown to a user but not necessarily clicked on. These can provide valuable negative feedback about items the user doesn't show interest in.

:p What role do impressions play in recommendation systems?
??x
Impressions help fill gaps by providing information on items that users don't interact with, thus offering negative feedback for improving recommendations.
x??

---

#### Event Collection and Instrumentation

Background context: Web applications often use events to log user interactions. Events are special messages generated when specific actions occur, such as a click or the addition of an item to a cart.

:p What is event collection in web applications?
??x
Event collection involves logging specially formatted messages that record user interactions like clicks and add-to-cart actions, providing detailed insights for analysis.
x??

---

#### Funnel Analyses

Background context: Funnels are used to track steps a user takes through a process. They help identify drop-off rates at each stage, which is crucial for evaluating the effectiveness of recommendation systems.

:p What is a funnel in data science?
??x
A funnel is a series of steps a user must take from one state to another, where users may "drop off" at any point, reducing the population size. It's used to track user journey stages.
x??

---

#### Types of Funnel Analyses

Background context: There are three key funnel analyses mentioned—page view to add-to-bag flow, page view to add-to-bag per recommendation, and add-to-bag to purchase completion.

:p What are the three main funnels discussed in the text?
??x
The three main funnels are:
1. Page view to add-to-bag user flow.
2. Page view to add-to-bag per recommendation.
3. Add-to-bag to complete purchase.
x??

---

#### Real-Time Handling with Event Streams

Background context: Event streams can be used for real-time handling of events, sending data to multiple destinations that can process it further.

:p How do event streams work in web applications?
??x
Event streams are used to send logged events to various downstream systems for processing. They often use technologies like Apache Kafka and can handle large volumes of data in real time.
x??

---

#### Sequence-Based Recommendations

Background context: Modern recommendation systems leverage the order of items a user clicks on, improving predictive accuracy.

:p What is sequence-based recommendation?
??x
Sequence-based recommendations utilize the order in which users click items to improve prediction accuracy. This approach has shown significant improvements by considering temporal patterns.
x??

---

