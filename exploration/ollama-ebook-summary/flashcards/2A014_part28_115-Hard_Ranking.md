# Flashcards: 2A014 (Part 28)

**Starting Chapter:** 115-Hard Ranking

---

#### Hard Avoids in Recommendation Systems
Background context explaining the concept. In recommendation systems, hard avoids are business rules that explicitly prevent certain items from being recommended to users, regardless of their preferences or collaborative filtering outcomes.

If a user has a strong dislike for an item (like grapefruit), it’s often more straightforward and consistent to remove all such items from recommendations rather than trying to learn this exception as part of the model. This approach ensures that negative experiences are mitigated directly in the system, avoiding potential ranking issues or model complexity.

:p How does the concept of hard avoids work in recommendation systems?
??x
Hard avoids involve explicitly removing certain items (like grapefruit) from a user's recommended list based on known business rules, rather than trying to learn exceptions through latent features. This method ensures that negative experiences are directly mitigated without overcomplicating the model.

```java
// Example of implementing hard avoids in Java pseudocode
public class RecommendationSystem {
    private Set<String> hardAvoids = new HashSet<>(Arrays.asList("grapefruit", "grapefruit cocktails"));

    public List<String> recommendItems(User user) {
        List<String> recommendedItems = fetchRecommendedItems(user);
        
        // Remove items from the recommendation list if they are in the hard avoids set
        return recommendedItems.stream()
                               .filter(item -> !hardAvoids.contains(item))
                               .collect(Collectors.toList());
    }
}
```
x??

---

#### Business Logic and Recommendation Systems
Background context explaining the complexity of integrating business logic into recommendation systems. While algorithmic ranking can provide highly personalized recommendations, real-world scenarios often require additional business rules to handle exceptions.

For instance, in a recipe recommendation system, even if a user likes most ingredients that pair well with grapefruit, the system should avoid recommending any recipes containing grapefruit due to the user’s strong dislike for it. Traditional collaborative filtering methods might not effectively address such specific dislikes.

:p How does the business logic intersect with recommendation systems?
??x
Business logic intersects with recommendation systems by implementing explicit rules or exceptions that are hard-coded into the recommendation process. This ensures that certain items, even if they would be recommended based on collaborative filtering or latent features, are excluded from the user's list to prevent negative user experiences.

```java
// Example of integrating business logic in Java pseudocode
public class RecipeRecommendationSystem {
    private Set<String> hardAvoids = new HashSet<>(Arrays.asList("grapefruit", "grapefruit cocktails"));

    public List<String> recommendRecipes(User user) {
        // Assume fetchRecommendedRecipes() is a method that returns a list of recommended recipes
        List<String> recommendedRecipes = fetchRecommendedRecipes(user);
        
        // Apply business logic to exclude hard avoids from the recommendation list
        return recommendedRecipes.stream()
                                 .filter(recipe -> !hardAvoids.contains(recipe))
                                 .collect(Collectors.toList());
    }
}
```
x??

---

#### Example of Hard Avoid in Recipe Recommendation System
Background context explaining how a specific instance (grapefruit) can be used to illustrate hard avoid concepts. In the given example, if a user strongly dislikes grapefruit but likes other ingredients that pair well with it, the recommendation system should not include any recipes containing grapefruit.

:p How does the concept of hard avoids apply in a recipe recommendation scenario?
??x
In a recipe recommendation scenario, hard avoids ensure that items like grapefruit are completely excluded from recommendations even if they would be recommended based on collaborative filtering or latent features. This is crucial for maintaining user satisfaction and preventing negative experiences due to known dislikes.

```java
// Example of implementing hard avoid in a recipe recommendation system
public class RecipeRecommendationSystem {
    private Set<String> hardAvoids = new HashSet<>(Arrays.asList("grapefruit", "grapefruit cocktails"));

    public List<String> recommendRecipes(User user) {
        // Assume fetchRecommendedRecipes() is a method that returns a list of recommended recipes
        List<String> recommendedRecipes = fetchRecommendedRecipes(user);
        
        // Apply business logic to exclude hard avoids from the recommendation list
        return recommendedRecipes.stream()
                                 .filter(recipe -> !hardAvoids.contains(recipe))
                                 .collect(Collectors.toList());
    }
}
```
x??

---

#### Hard Ranking
Background context explaining hard ranking. Hard ranking usually refers to one of two kinds of special ranking rules: explicitly removing some items from the list before ranking or using a categorical feature to rank results by category, even for multiple features to achieve hierarchical hard ranking.

:p Can you provide examples of hard ranking?
??x
Examples include:
- A user bought a sofa and the system continues to recommend sofas despite it being unlikely they will need another one.
- An e-commerce site keeps recommending gardening tools after a user buys gifts related to gardening, even though the user has no interest in it anymore.
- A website recommends toys for an older child when the parent hasn't purchased from them since the child was younger.
- Meetup recommendations are still all running oriented despite someone switching to cycling due to knee pain.

The system should respect these explicit expectations by not showing irrelevant items, as learning such rules via ML would be difficult and may harm user satisfaction. These hard ranking rules can be implemented deterministically with simple logic or models during the serving stage.
x??

---

#### Learned Avoids
Background context explaining learned avoids. Not all business rules are obvious like explicit user feedback. Some arise from indirect feedback, such as ignored categories or low-quality items.

:p Can you provide examples of learned avoids?
??x
Examples include:
- Already owned items: Recommending clothing the user has already purchased.
- Disliked features: Avoiding certain item features based on user preferences expressed in an onboarding questionnaire.
- Ignored categories: Not showing items from categories that users do not engage with, such as dresses if they never click them.

These avoids can be implemented easily during serving and can improve ranking performance. Training simple models to detect these rules and applying them during serving is a useful mechanism for improving recommendation quality.
x??

---

#### Hand-Tuned Weights
Background context explaining hand-tuned weights. Hand-tuned ranking was popular in early search ranking, where humans used analytics and observation to determine important features. However, such approaches don't scale well.

:p What is an example of using hand-tuned weights as an avoid?
??x
An example is up-ranking lower-priced items before a user's first order to build trust in the shipping process. A fashion recommender system could use a style expert's knowledge about summer trends (e.g., mauve colors) and rank such items higher for users who fit that age persona.

While it may feel bad to consider building hand-tuned ranking, this technique can be useful as an avoid when used carefully.
x??

---

#### Inventory Health
Background context explaining inventory health. Inventory health estimates how good the current stock is at satisfying user demand. It involves leveraging forecasts and tracking actual sales to optimize inventory levels.

:p How can you define inventory health?
??x
Inventory health can be defined using affinity scores and forecasting. By analyzing demand forecasts, you can estimate expected sales in each category over time periods. This helps determine how many items of various types should be kept on hand.

For example, if the forecast predicts high demand for poppy seed bagels, ensuring sufficient stock is crucial to avoid disappointing customers who want those specific types.
x??

---

#### Implementing Avoids
Background context explaining implementing avoids via downstream filtering. Filtering rules are applied before recommendations reach users to ensure they don't violate explicit or learned preferences.

:p How can you implement avoiding already owned items using pandas?
??x
To filter out already owned items, you would use a function like this:
```python
import pandas as pd

def filter_dataframe(df: pd.DataFrame, filter_dict: dict):
    """Filter dataframe to exclude rows where columns have certain values."""
    for col, val in filter_dict.items():
        df = df.loc[df[col] != val]
    return df

filter_dict = {'column1': 'value1', 'column2': 'value2', 'column3': 'value3'}
df = df.pipe(filter_dataframe, filter_dict)
```

This is a basic approach but can be improved using more powerful tools like JAX for scalability.
x??

---

#### Implementing Avoids with JAX
Background context explaining implementing avoids using JAX. JAX allows for efficient array manipulation and filtering.

:p How can you implement avoiding already owned items using JAX?
??x
To filter out already owned items in JAX, you would use a function like this:
```python
import jax
import jax.numpy as jnp

def filter_jax_array(arr: jnp.array, col_indices: list, values: list):
    """Filter array to exclude rows where certain columns have certain values."""
    assert len(col_indices) == len(values),
    masks = [arr[:, col] != val for col, val in zip(col_indices, values)]
    total_mask = jnp.logical_and(*masks)
    return arr[total_mask]
```

This function applies logical AND across multiple conditions to filter the array. It's more scalable than pandas but still needs further optimization.
x??

---

#### Conditional Avoids
Background context explaining conditional avoids. Some avoid rules depend on contextual factors and may require explicit specification.

:p Can you provide an example of a conditional avoid?
??x
An example is not wearing white after Labor Day or avoiding meat on Fridays based on user preferences. Coffee-processing methods that don't mesh well with certain brewers are another case where conditional logic is needed to ensure the best recommendation.

These types of rules often have lower signal compared to broader trends and should be explicitly specified rather than relying on ML models to learn them.
x??

---

#### Feature Stores for Avoids
Background context explaining feature stores as a repository for avoid rules. Feature stores can store avoids in real-time or upon user onboarding, making them easy to retrieve.

:p How can you use feature stores to manage avoid rules?
??x
Feature stores allow storing and retrieving avoid rules efficiently. You can key these rules by user ID and fetch them during the serving process. For example:
```python
# Example of fetching avoids from a feature store
def get_user_avoids(user_id):
    # Fetch user-specific avoids from a feature store
    avoids = feature_store.get(user_id)
    return avoids

user_avoids = get_user_avoids('user123')
```

This approach ensures that the system can quickly access and apply avoid rules tailored to each user.
x??


--- 
Note: Ensure all code examples are correct and relevant. The provided prompts and answers should be clear and concise, focusing on key concepts rather than unnecessary details. Use JAX or pandas as appropriate based on context. ---

#### Explicit Deterministic Algorithms for Coffee Problem

Background context: The specification can be achieved using explicit deterministic algorithms that impose specific requirements. For instance, a decision stump was hand-built to handle bad combinations of coffee roast features and brewers.

:p What is an example of how explicit deterministic algorithms are used in the coffee problem?
??x
An example involves creating a simple decision rule or tree (decision stump) to manage specific undesirable pairings between coffee roast features and brewing methods. For instance, avoiding "anaerobic espresso" which was deemed unacceptable by one of the authors.
x??

---

#### Nuanced Examples: Not Wearing White After Labor Day

Background context: Some rules are more nuanced and difficult to implement with explicit algorithms. The example given is not wearing white after Labor Day, where determining if a user wears white on that day can be tricky.

:p How do we handle nuanced rules like "not wearing white after Labor Day" using an algorithm?
??x
Handling nuanced rules such as "not wearing white after Labor Day" requires more sophisticated methods. These rules are often approached through model-based avoids, where simple regression models or latent feature representations are used to infer user behavior and preferences.
x??

---

#### Model-Based Avoids for Friday Vegetarians

Background context: For more complex rules like not eating meat on Fridays, model-based avoids can be effective. This involves inferring a persona that has this rule from other attributes.

:p How do we use latent feature representations to handle the "not eating meat on Fridays" rule?
??x
To handle the "not eating meat on Fridays" rule, we infer a persona through latent feature modeling. This involves mapping known attributes to infer the persona and then using regression models to find features that correlate with this inferred persona.

```java
// Pseudocode for inferring vegetarian personas
public class VegetarianPersonaInference {
    private Map<String, Double> userAttributes;
    
    public void inferPersonas() {
        // Model latent features from user attributes
        // This step maps known attributes to potential personas
        // userAttributes -> inferredPersonas
        
        // Use regression models to find correlation between inferred personas and eating habits on Fridays
        for (String feature : userAttributes.keySet()) {
            double correlation = calculateCorrelation(feature, "not eating meat on Fridays");
            if (correlation > threshold) {
                System.out.println("User likely has a vegetarian Friday persona: " + feature);
            }
        }
    }
    
    private double calculateCorrelation(String feature, String behavior) {
        // Calculate the correlation between the user attribute and the behavior
        return 0.5; // Dummy value for illustration
    }
}
```
x??

---

#### Implementing Avoids in Recommendation Systems

Background context: Ensuring that recommendations satisfy essential business rules can be achieved using both explicit deterministic algorithms and model-based avoids. The implementation is often complex, requiring a balance between relevance and diversity.

:p How do we implement ensures (business logic) in recommendation systems?
??x
To implement ensures in recommendation systems, you might use explicit deterministic algorithms for simple cases or model-based avoids for more nuanced rules. For instance, using decision stumps for specific combinations of features can be effective. More complex rules like "not wearing white after Labor Day" are better handled through latent feature modeling and regression.

```java
// Example implementation using a rule-based approach
public class EnsureRecommendations {
    private Map<String, Boolean> businessRules;
    
    public void applyBusinessRules(String user, List<Item> items) {
        for (Item item : items) {
            if (!businessRules.get(user + ":" + item)) {
                // Exclude the item based on the rule
                System.out.println("Excluding " + item + " for user " + user);
            }
        }
    }
}
```
x??

---

#### Diversity in Recommendation Outputs

Background context: Ensuring a variety of recommendations is crucial to prevent overspecialization and promote novel discoveries. This involves balancing relevance with diversity.

:p How do we balance relevance and diversity in recommendation outputs?
??x
Balancing relevance and diversity requires careful consideration of the user's preferences while also exposing them to new content. This can be achieved by using a combination of personalized recommendations and diversification techniques such as random sampling or collaborative filtering to introduce variety.

```java
// Pseudocode for balancing relevance and diversity
public class RecommendationService {
    private List<Item> relevantItems;
    private List<Item> diverseItems;
    
    public void serveRecommendations(String userId) {
        int numRelevant = 3; // Number of highly relevant items
        int numDiverse = 2; // Number of diverse items to introduce variety
        
        for (int i = 0; i < numRelevant; i++) {
            Item item = findMostRelevantItem(userId);
            System.out.println("Recommend " + item + " to user " + userId);
        }
        
        while (numDiverse > 0) {
            Item diverseItem = getRandomDiverseItem();
            if (!relevantItems.contains(diverseItem)) {
                relevantItems.add(diverseItem);
                numDiverse--;
                System.out.println("Recommend " + diverseItem + " for variety");
            }
        }
    }
    
    private Item findMostRelevantItem(String userId) {
        // Logic to find the most relevant item
        return null; // Dummy value for illustration
    }
    
    private Item getRandomDiverseItem() {
        // Logic to get a random diverse item
        return null; // Dummy value for illustration
    }
}
```
x??

---

#### Bias in Recommendation Systems

Background context: Recommendation systems can suffer from biases, including overly redundant or self-similar sets of recommendations and stereotypes learned by AI. Ensuring diversity helps mitigate these issues.

:p What are the two most important kinds of bias in recommendation systems?
??x
The two most important kinds of bias in recommendation systems are:
1. Overly redundant or self-similar sets of recommendations.
2. Stereotypes learned by AI systems.

These biases can lead to overspecialization and perpetuation of stereotypes, reducing the overall quality of recommendations.
x??

---

