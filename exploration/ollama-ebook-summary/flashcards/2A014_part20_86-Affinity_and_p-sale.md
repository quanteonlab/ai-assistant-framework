# Flashcards: 2A014 (Part 20)

**Starting Chapter:** 86-Affinity and p-sale

---

#### Centered Kernel Alignment
Centered Kernel Alignment is a technique used to compare layer representations within neural networks. This method helps in understanding how these latent space representations change and correlate across different layers, providing insights into the network's behavior.
:p How does Centered Kernel Alignment help in comparing layer representations in neural networks?
??x
Centered Kernel Alignment aids in analyzing the correlation structures between incoming signals at each layer by representing them as a sequence of states. By comparing these latent space representations across layers using an N×N matrix, where N is the number of layers, one can understand how similar or different these layers are and gain insights into the network's internal functioning.
??x

---

#### Affinity and p-sale
Affinity and p-sale are key concepts in Matrix Factorization (MF). The affinity score represents the similarity between a user's preferences and a product’s characteristics. However, this score alone may not accurately predict whether a sale will occur due to various external factors.

The probability of a sale is estimated using a logistic function applied to the dot product of the corresponding row in the user matrix and column in the product matrix.
:p What does affinity and p-sale represent in Matrix Factorization?
??x
Affinity represents the similarity between a user's preferences (encoded as a vector) and a product’s characteristics (also encoded as a vector). It is calculated using the dot product of these vectors. However, while this score indicates how well the user’s preferences align with the product’s characteristics, it may not be sufficient to predict whether a sale will actually occur.

The p-sale, or probability of sale, is derived from the affinity score by applying a logistic function (sigmoid). This transformation takes into account additional factors such as the overall popularity of the product and the user's purchasing behavior. The formula for calculating the p-sale is:

\[ P(u, p) = \text{sigmoid}(u^T p) \]

Where \( u \) is the row vector representing a user’s preferences, \( p \) is the column vector representing a product’s characteristics, and \( \text{sigmoid} \) is the logistic function defined as:
\[ \text{sigmoid}(x) = \frac{1}{1 + e^{-x}} \]

:p What is the formula for calculating the probability of sale in Matrix Factorization?
??x
The formula for calculating the probability of sale (p-sale) in Matrix Factorization is:

\[ P(u, p) = \text{sigmoid}(u^T p) \]

Where \( u \) represents a user's preferences and \( p \) represents a product’s characteristics. The dot product \( u^T p \) measures the similarity between the vectors, and the sigmoid function maps this similarity score to a probability score between 0 and 1.

The code for applying the logistic function in Java could look like this:
```java
public class LogisticFunction {
    public static double sigmoid(double x) {
        return 1 / (1 + Math.exp(-x));
    }
}
```
This function takes the dot product as input and returns a probability score.
??x

---

#### Alternating Least Squares (ALS)
Alternating Least Squares is an optimization algorithm used to train matrices in Matrix Factorization. It alternates between fixing one matrix while solving for the other, ensuring fast convergence due to its convex loss function.

The ALS method works by iteratively solving for the user and product matrices until convergence.
:p How does Alternating Least Squares (ALS) work?
??x
Alternating Least Squares (ALS) is an optimization algorithm used in Matrix Factorization to train two matrices: one representing users' preferences (\( U \)) and another representing products' characteristics (\( V \)). The method alternates between fixing the user matrix and solving for the product matrix, then fixing the product matrix and solving for the user matrix.

The ALS loss function is convex, meaning there is a single global minimum. This property allows for fast convergence when either matrix is fixed during each iteration. Here’s an example of how it works:

1. Initialize \( U \) and \( V \).
2. Fix \( V \), solve for \( U \).
3. Fix \( U \), solve for \( V \).
4. Repeat steps 2-3 until convergence.

The code for a single iteration in Java might look like this:
```java
public class ALS {
    public static void train(Matrix R, int rank) {
        Matrix U = initializeU(R.numRows(), rank);
        Matrix V = initializeV(R.numCols(), rank);

        while (!converged(U, V)) {
            // Fix V and solve for U
            for (int i = 0; i < U.rows(); ++i) {
                double[] ui = U.getRow(i).toArray();
                for (int j = 0; j < V.cols(); ++j) {
                    double[] vjT = V.getCol(j).toArray();
                    // Update ui based on the current V and R
                    // This involves solving a system of equations or using gradient descent
                }
            }

            // Fix U and solve for V
            for (int j = 0; j < V.cols(); ++j) {
                double[] vj = V.getRow(j).toArray();
                for (int i = 0; i < U.rows(); ++i) {
                    double[] uiT = U.getCol(i).toArray();
                    // Update vj based on the current U and R
                    // This involves solving a system of equations or using gradient descent
                }
            }
        }
    }

    private static boolean converged(Matrix U, Matrix V) {
        // Check for convergence criteria (e.g., small change in matrix values)
        return true;
    }

    private static Matrix initializeU(int rows, int rank) {
        // Initialize U with random values or some heuristic
        return new Matrix(rows, rank);
    }

    private static Matrix initializeV(int cols, int rank) {
        // Initialize V with random values or some heuristic
        return new Matrix(cols, rank);
    }
}
```
This code provides a simplified outline of how ALS works in an iterative manner.
??x

---

#### Factorization Machines (FM)
Factorization Machines are a general model that can be used for regression and binary classification tasks. They factorize the dot product between two vectors to capture interactions between features.

The FM model includes both main effects (individual feature effects) and interaction effects, making it more flexible than simple linear models.
:p What is Factorization Machines (FM)?
??x
Factorization Machines are a general machine learning model that can handle both regression and binary classification tasks. They factorize the dot product between two vectors to capture interactions between features, which makes them more expressive than simple linear models.

The FM model includes main effects for individual feature values and interaction effects that capture the combined influence of pairs or higher-order combinations of features.

The formulation for a Factorization Machine is similar to GloVe embeddings in that it models the interaction between two vectors. However, instead of explicitly modeling all interactions, FM uses factorized vectors to represent these interactions.

:p What are the key components of Factorization Machines?
??x
Factorization Machines consist of three main components:

1. **Main Effects**: These capture the individual effect of each feature.
2. **Interaction Effects**: These model the interaction between pairs of features.
3. **Linear and Non-linear Terms**: The non-linear terms allow for more complex interactions.

The overall prediction \( P \) can be expressed as:
\[ P(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Where:
- \( w_0 \) is the bias term.
- \( w_i \) are the main effect weights for feature \( i \).
- \( v_i \) are the factorized vectors representing the interaction between features.
- \( \langle v_i, v_j \rangle \) is the dot product of the factorized vectors.

:p How does Factorization Machines model interactions?
??x
Factorization Machines model interactions by using factorized vectors to represent the combined influence of pairs or higher-order combinations of features. This allows for a more expressive and flexible representation compared to simple linear models.

The interaction term in the FM model is given by:
\[ \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Where \( v_i \) and \( v_j \) are factorized vectors for features \( i \) and \( j \), respectively, and \( x_i \) and \( x_j \) are the values of these features. The dot product \( \langle v_i, v_j \rangle \) captures the interaction between features.

:p What is the formula for the prediction in Factorization Machines?
??x
The overall prediction in a Factorization Machine can be expressed as:
\[ P(x) = w_0 + \sum_{i=1}^{n} w_i x_i + \sum_{i=1}^{n} \sum_{j=i+1}^{n} \langle v_i, v_j \rangle x_i x_j \]

Where:
- \( w_0 \) is the bias term.
- \( w_i \) are the main effect weights for feature \( i \).
- \( v_i \) are the factorized vectors representing the interaction between features.
- \( \langle v_i, v_j \rangle \) is the dot product of the factorized vectors.

This formula includes both main effects and interaction effects, making FM a flexible model that can handle complex interactions between features.

#### Feedback Loop and Causal Inference Challenges

Background context: Recommendation systems are evaluated using user feedback, but this data is causally influenced by the deployed system. This creates a feedback loop that can introduce bias in model evaluation.

:p What challenges does the feedback loop present for evaluating recommendation systems?
??x
The feedback loop presents several challenges:
1. **User Bias**: Users' actions and outcomes are influenced by the recommendations they receive, making it difficult to distinguish between user preferences and system influence.
2. **Confounding Variables**: The deployed system can act as a confounder, complicating the evaluation of new models since their performance is intertwined with the system's existing behavior.

Code Example:
```java
// Pseudocode for modeling feedback loop
public class FeedbackLoop {
    private RecommendationSystem deployedSystem;
    
    public void evaluateNewModel(Recommender newModel) {
        // Collect user interactions and feedback
        List<UserInteraction> interactions = deployedSystem.getInteractions();
        
        // Evaluate the performance of the new model using these interactions
        double performance = evaluatePerformance(interactions, newModel);
        
        System.out.println("Performance: " + performance);
    }
    
    private double evaluatePerformance(List<UserInteraction> interactions, Recommender newModel) {
        int totalUsers = interactions.size();
        int satisfiedUsers = 0;
        
        for (UserInteraction interaction : interactions) {
            if (newModel.recommend(interaction.user).contains(interaction.item)) {
                satisfiedUsers++;
            }
        }
        
        return (double) satisfiedUsers / totalUsers;
    }
}
```
x??

---

#### Propensity Score and Propensity Weighting

Background context: Propensity scores are used to adjust for the bias introduced by the deployed recommendation system. The propensity score quantifies the likelihood of a user seeing an item.

:p What is the purpose of using propensity weighting in evaluating recommendation systems?
??x
The purpose of using propensity weighting in evaluating recommendation systems is to mitigate the bias caused by the closed-loop feedback loop and to provide a more accurate evaluation of new models. By adjusting for the probability that an item was shown to a user, propensity weighting helps account for the influence of the deployed system on user interactions.

Code Example:
```java
// Pseudocode for calculating propensity scores
public class PropensityScoring {
    private Map<Item, Double> propensityScores;
    
    public void estimatePropensities(Map<User, Set<Item>> interactions) {
        // Estimate propensity scores using maximum likelihood or other methods
        for (User user : interactions.keySet()) {
            for (Item item : interactions.get(user)) {
                double propensity = calculatePropensity(item, user);
                propensityScores.put(item, propensity);
            }
        }
    }
    
    private double calculatePropensity(Item item, User user) {
        // Simplified example
        return Math.random();  // In practice, use more sophisticated methods
    }
}
```
x??

---

#### Simpson’s Paradox and Confounding Variables

Background context: Simpson's paradox occurs when the relationship between two variables appears to change direction when examined within different strata. This is relevant in recommendation systems where the deployed model’s characteristics can create biased feedback.

:p How does Simpson’s paradox relate to confounding variables in recommendation systems?
??x
Simpson’s paradox relates to confounding variables in recommendation systems by highlighting how a single overall trend may mask underlying biases when data is stratified. In this context, the deployed model's characteristics act as a confounder that can create misleading correlations between user interactions and recommendations.

Code Example:
```java
// Pseudocode for demonstrating Simpson’s paradox
public class SimpsonParadox {
    private Map<Item, Double> propensityScores;
    
    public void simulateSimpsonParadox() {
        // Simulate data where the overall trend is positive but within strata it's negative
        List<UserInteraction> interactions = new ArrayList<>();
        
        for (int i = 0; i < 1000; i++) {
            User user = new User("User" + i);
            Item item = new Item("Item" + i);
            
            if ((i % 2 == 0) && Math.random() > 0.5) {
                interactions.add(new UserInteraction(user, item));
            }
        }
        
        // Estimate propensity scores
        estimatePropensities(interactions);
    }
    
    private void estimatePropensities(List<UserInteraction> interactions) {
        for (UserInteraction interaction : interactions) {
            double propensity = calculatePropensity(interaction.item, interaction.user);
            propensityScores.put(interaction.item, propensity);
        }
    }
    
    private double calculatePropensity(Item item, User user) {
        // Simplified example
        return Math.random();  // In practice, use more sophisticated methods
    }
}
```
x??

---

#### Inverse Propensity Scoring (IPS)

Background context: IPS is a method used to evaluate recommendation systems by accounting for the non-uniform exposure of items due to the deployed system. It uses importance sampling to reweight feedback.

:p What is inverse propensity scoring (IPS) and how does it address issues in evaluating recommendation systems?
??x
Inverse Propensity Scoring (IPS) addresses evaluation issues in recommendation systems by using importance sampling to adjust for the non-uniform exposure of items due to the deployed system. It reweights user interactions based on the probability that an item was shown to a user, thereby providing a more accurate reflection of how well new models perform.

Code Example:
```java
// Pseudocode for IPS evaluation
public class IPSEvaluation {
    private Map<Item, Double> propensityScores;
    
    public void evaluateRecommender(Recommender model) {
        // Collect feedback from deployed system
        List<UserInteraction> interactions = deployedSystem.getInteractions();
        
        double totalWeightedPerformance = 0.0;
        int numItemsEvaluated = 0;
        
        for (UserInteraction interaction : interactions) {
            double propensity = propensityScores.getOrDefault(interaction.item, 1.0);
            double weight = 1.0 / propensity;  // Inverse of the propensity score
            totalWeightedPerformance += model.recommend(interaction.user).contains(interaction.item) ? 1.0 * weight : 0;
            numItemsEvaluated++;
        }
        
        double averagePerformance = totalWeightedPerformance / numItemsEvaluated;
        System.out.println("Average Performance (IPS): " + averagePerformance);
    }
}
```
x??

---

#### Doubly Robust Estimation

Doubly robust estimation (DRE) is a method that combines two models: one that models the probability of receiving the treatment (being recommended an item by the deployed model) and one that models the outcome of interest (the user’s feedback on the item). The weights used in DRE depend on the predicted probabilities from both models. This method has the advantage that it can still provide unbiased estimates even if one of the models is misspecified.

The structural equations for a doubly robust estimator with propensity score weighting and outcome model are as follows:

\[ \Theta = \sum w_i Y_i - f(X) + \sum w_ip_i (1 - p_i) f^*(X_i) - f^* (X_i) \]

Where:
- \(Y_i\) is the outcome,
- \(X_i\) are covariates,
- \(T_i\) is the treatment,
- \(p_i\) is the propensity score,
- \(w_i\) is the weight,
- \(f(X)\) is the outcome model, and
- \(f^*(X)\) is the estimated outcome model.

:p What is doubly robust estimation (DRE)?
??x
Double robust estimation combines two models: one for predicting the probability of receiving treatment and another for modeling the outcome. It ensures unbiased estimates even if one of these models is misspecified.
x??

---

#### Latent Spaces in Recommendation Systems

Latent spaces are a critical aspect of recommendation systems, representing users and items through encoded representations. Beyond dimension reduction, latent spaces help capture meaningful relationships that inform the ML task.

:p What are latent spaces?
??x
Latent spaces encode users and items to represent their underlying features or preferences, enabling more effective recommendation by understanding the geometric structure.
x??

---

#### Personalized Recommendation Metrics

Key metrics for evaluating personalized recommendations include mean average precision (mAP), mean reciprocal rank (MRR), and normalized discounted cumulative gain (NDCG). These metrics assess different aspects of user interaction with recommended items.

:p What are some key ranking metrics in recommendation systems?
??x
Mean Average Precision (mAP), Mean Reciprocal Rank (MRR), and Normalized Discounted Cumulative Gain (NDCG) are essential metrics for evaluating the quality of ranked recommendations.
x??

---

#### mAP, MRR, NDCG Metrics

- **mAP** measures average precision at different recall levels.
- **MRR** ranks items based on their reciprocal rank position in the list.
- **NDCG** discounts irrelevant items and normalizes gains to evaluate relevance.

:p What are the differences between mAP, MRR, and NDCG?
??x
Mean Average Precision (mAP) assesses average precision at various recall levels. Mean Reciprocal Rank (MRR) focuses on the position of relevant items in the ranking list. Normalized Discounted Cumulative Gain (NDCG) evaluates relevance by discounting irrelevant items and normalizing gains.
x??

---

#### Example Code for Ranking Evaluation

```java
public class RecommendationMetrics {
    public double calculateMAP(List<RelevanceScore> scores, int k) {
        // Implementation of mAP calculation
    }

    public double calculateMRR(List<RelevanceScore> scores) {
        // Implementation of MRR calculation
    }

    public double calculateNDCG(List<RelevanceScore> scores, int k) {
        // Implementation of NDCG calculation
    }
}

class RelevanceScore {
    private final String item;
    private final double relevance;

    public RelevanceScore(String item, double relevance) {
        this.item = item;
        this.relevance = relevance;
    }

    // Getter methods for item and relevance
}
```

:p How would you implement mAP, MRR, and NDCG in Java?
??x
You can implement these metrics using a class like `RecommendationMetrics` with methods to calculate each metric. For example:

```java
public class RecommendationMetrics {
    public double calculateMAP(List<RelevanceScore> scores, int k) {
        // Implementation of mAP calculation
        return 0; // Placeholder
    }

    public double calculateMRR(List<RelevanceScore> scores) {
        // Implementation of MRR calculation
        return 0; // Placeholder
    }

    public double calculateNDCG(List<RelevanceScore> scores, int k) {
        // Implementation of NDCG calculation
        return 0; // Placeholder
    }
}

class RelevanceScore {
    private final String item;
    private final double relevance;

    public RelevanceScore(String item, double relevance) {
        this.item = item;
        this.relevance = relevance;
    }

    // Getter methods for item and relevance
}
```
x??

---

