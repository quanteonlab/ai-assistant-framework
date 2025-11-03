# Flashcards: 2A014 (Part 33)

**Starting Chapter:** 131-Summary

---

#### Transformers4Rec
Background context: `Transformers4Rec` is an open-source project developed by NVIDIA's Merlin team, focusing on scalable transformer models for recommendation systems. This approach bridges the gap between natural language processing (NLP) and sequential or session-based recommendation tasks.

:p What is `Transformers4Rec` used for?
??x
`Transformers4Rec` is designed to handle large-scale recommendation datasets by employing scalable transformer architectures, which are originally developed for NLP tasks. This project aims to leverage the power of transformers to improve the accuracy and efficiency of recommendations in sequential or session-based scenarios.
```java
// Pseudocode for initializing a basic Transformer model
public class Transformers4Rec {
    private List<EmbeddingLayer> embeddingLayers;
    private TransformerEncoder transformerEncoder;

    public Transformers4Rec(List<String> features) {
        this.embeddingLayers = new ArrayList<>();
        // Initialize embedding layers for each feature
        this.transformerEncoder = new TransformerEncoder();
    }

    public void train(List<Samples> data) {
        // Training logic goes here
    }
}
```
x??

---

#### Monolith Recommender System
Background context: `Monolith` is a real-time recommendation system used by TikTok for user page recommendations. It is known for its elegant hybrid approaches and is one of the most popular and exciting recommendation systems at this time.

:p What is `Monolith`?
??x
`Monolith` refers to a sophisticated real-time recommendation system developed by TikTok, utilizing hybrid approaches that combine different recommendation techniques. This system is designed to provide personalized recommendations in near-real-time.
```java
// Pseudocode for Monolith's architecture
public class Monolith {
    private CollisionlessEmbeddingTable embeddingTable;
    private RealTimeRecommendationEngine engine;

    public Monolith(CollisionlessEmbeddingTable embeddingTable) {
        this.embeddingTable = embeddingTable;
        this.engine = new RealTimeRecommendationEngine();
    }

    public List<RecommendedItem> getRecommendations(UserProfile userProfile, int topN) {
        // Logic to fetch and recommend items
        return engine.getTopNItems(userProfile, topN);
    }
}
```
x??

---

#### Multimodal Recommendations
Background context explaining multimodal recommendations. Users have diverse preferences that can be represented by several latent vectors simultaneously.

:p What is a multimodal recommendation?
??x
Multimodal recommendations recognize that users may have multiple and conflicting interests. For example, someone shopping on an e-commerce site could be:
- A dog owner who frequently needs items for their dog
- A parent updating the closet for a growing baby
- A hobbyist race-car driver buying parts
- An investor in LEGO sets

This leads to multimodal preferences: multiple latent factors coalesce into modes or medoids, rather than one. Nearest neighbors may struggle to find relevant recommendations if the user's interests are conflicting.

??x
The answer with detailed explanations.
```java
// Example of handling multimodal user vectors in Java
public class UserVector {
    private Map<String, Double[]> modalVectors; // key is mode name, value is vector

    public UserVector(Map<String, Double[]> modalVectors) {
        this.modalVectors = modalVectors;
    }

    public double[] getModalVector(String modeName) {
        return modalVectors.get(modeName);
    }
}
```
x??

---

#### PinnerSage
Background context explaining the PinnerSage approach. It uses clustering to build modes in item space, allowing for more flexible representation of users' diverse interests.

:p What is PinnerSage and how does it work?
??x
PinnerSage is a multimodal recommender system that clusters user interactions (unsupervised) to form cluster representations as medoids. It uses graph-based feature representations and aims to build modes via clustering in item space. Key steps include:
1. Fixing item embeddings ("pins").
2. Clustering user interactions.
3. Building cluster representations as the medoid of the cluster embeddings.
4. Retrieving using medoid-anchored ANN search.

??x
The answer with detailed explanations.
```java
// Example of PinnerSage clustering in Java
public class PinClusterer {
    private List<PinnedItem> pins; // item embeddings

    public PinClusterer(List<PinnedItem> pins) {
        this.pins = pins;
    }

    public void cluster() {
        // Cluster user interactions to form medoids of clusters
    }

    public PinnedItem getMedoid(int clusterId) {
        // Return the medoid of a given cluster
        return pins.get(clusterId);
    }
}
```
x??

---

#### Graph Neural Networks (GNNs)
Background context explaining GNNs and their application in recommendation systems. They use structural information to build deeper representations, allowing for explicit representation of higher-order relationships.

:p What is a graph neural network (GNN) and how does it differ from traditional neural networks?
??x
Graph Neural Networks (GNNs) are a class of neural networks that utilize the structural information in data to build deeper representations. They are particularly useful for relational or networked data. The key difference between GNNs and traditional neural networks is during training, where explicit operators transfer data "along edges" via message passing.

GNNs work by assigning objects as nodes and relationships as edges. During training, a message function sends features from one node to another along the edge. An aggregation function then combines these messages into a single representation for each node or edge.

:p What is message passing in GNNs?
??x
Message passing in GNNs involves transferring data between node representations "along the edges" during training. This process allows GNNs to capture and utilize relationships within the graph structure.

Example of message function:
```java
public double[] sendMessage(double[] featuresI, double[] featuresJ, double[] edgeFeatures) {
    // Use differentiable function to combine features from nodes and edge
    return messageFunction(featuresI, featuresJ, edgeFeatures);
}
```

:p What are some common aggregation functions used in GNNs?
??x
Common aggregation functions in GNNs include:
- Concatenation: `concatenate all the messages`
- Summation: `sum all the messages`
- Averaging: `average all the messages`
- Max-pooling: `take the max of the messages`

:p What is an example of a simple message function?
??x
A simple example of a message function in GNNs could be:
```java
public double[] sendMessage(double[] featuresI, double[] featuresJ) {
    // Take the features from a neighbor node (no edge-specific info)
    return new double[]{featuresI[0], featuresJ[1]};
}
```
x??

---

#### Example of Graph-based Recommendation System
Background context explaining how higher-order relationships can be explicitly specified in recommendation systems using graphs.

:p How can higher-order relationships between items or users be represented in a graph?
??x
Higher-order relationships in graphs can be represented by adding structure to the basic node-edge framework. Examples include:
- Directionality: indicating strict relationships (e.g., user reads book, not vice versa).
- Edge decorations: adding features like edge labels (e.g., shared account credentials, where one is a child).
- Multiedges: allowing multiple relationships between the same entities.
- Hyper-edges: connecting multiple nodes simultaneously (e.g., detecting object classes and their combinations in video scenes).

:p How does GNNs use message passing for recommendation?
??x
GNNs use message passing to transfer data between node representations "along edges." This allows them to capture relationships within the graph structure. For example, in a social media context, features like demographic information could be used as node features, and friendships as edge features.

:p How does PinnerSage differ from traditional matrix factorization methods?
??x
PinnerSage differs from traditional matrix factorization methods by:
- Clustering user interactions to form medoids of clusters.
- Using graph-based feature representations.
- Attempting to build modes via clustering in item space, rather than a single latent factor.

:p How does PinnerSage retrieve recommendations?
??x
PinnerSage retrieves recommendations using medoid-anchored ANN search. This involves:
1. Fixing item embeddings ("pins").
2. Clustering user interactions.
3. Building cluster representations as the medoids of clusters.
4. Retrieving using the medoids.

??x
The answer with detailed explanations.
```java
// Example of PinnerSage recommendation retrieval in Java
public class PinRecommender {
    private Map<Integer, PinnedItem> pinMap; // map from user to cluster medoid

    public PinRecommender(Map<Integer, PinnedItem> pinMap) {
        this.pinMap = pinMap;
    }

    public List<PinnedItem> recommendForUser(int userId) {
        PinnedItem medoid = pinMap.get(userId);
        return medoid.retrieveSimilarPins(); // retrieve similar pins based on ANN search
    }
}
```
x??

---

#### Modeling User-Item Interactions
Background context explaining how traditional methods like matrix factorization handle user-item interactions, but do not fully utilize the complex network structure. GNNs can capture these connections to make more accurate recommendations.

:p How does GNN help model user-item interactions differently from other methods?
??x
GNN helps by capturing the complex relationships in the user-item interaction graph and using this structural information to generate more accurate recommendations. In contrast, traditional methods like matrix factorization treat each interaction as a simple point without leveraging the network structure.

```java
// Pseudocode for GNN Interaction Model
public class UserItemInteraction {
    public List<Node> getUsers() { ... }
    public List<Node> getItems() { ... }
    public Map<User, Set<Item>> getUserItemInteractions() { ... }
}
```
x??

---

#### Feature Learning in GNNs
Background context explaining that GNNs can learn more expressive feature representations of nodes by aggregating information from their neighbors. This allows the network to build a latent representation from messages passed between items and users.

:p How does GNN perform feature learning?
??x
GNN performs feature learning by aggregating feature information from neighboring nodes, thus leveraging the connections in the graph to provide rich information about user preferences or item characteristics. This process can be more powerful than other methods because it explicitly defines structural relationships and how they communicate features.

```java
// Pseudocode for Feature Aggregation in GNN
public class Node {
    public List<Node> getNeighbors() { ... }
    public void updateFeature(Map<Node, FeatureVector> neighborFeatures) { ... }
}
```
x??

---

#### Cold-Start Problem with GNNs
Background context explaining the challenge of providing recommendations to new users or items due to a lack of historical interactions. GNN can learn embeddings for new nodes using graph structure and node features.

:p How does GNN address the cold-start problem?
??x
GNN addresses the cold-start problem by learning embeddings for new users or items based on their features and the graph's structure, potentially alleviating issues where there is a lack of historical interactions. For example, structural edges like "share a physical location" can help quickly generate recommendations for new users.

```java
// Pseudocode for Cold-Start Solution with GNN
public class Node {
    public void learnEmbedding() { ... }
}
```
x??

---

#### Context-Aware Recommendations with GNNs
Background context explaining that GNNs can incorporate contextual information into the recommendation process, such as modeling a session's interaction sequence to make dynamic and complex recommendations.

:p How does GNN enable context-aware recommendations?
??x
GNN enables context-aware recommendations by incorporating sequential data from user interactions within a session. It models these interactions as a graph where nodes represent items and edges represent the order of interactions, allowing it to learn transitions and provide recommendations based on this context.

```java
// Pseudocode for Context-Aware Recommendations with GNN
public class Session {
    public List<Item> getItemsInSession() { ... }
    public void generateRecommendations(Node userNode) { ... }
}
```
x??

---

#### Random Walks in GNNs
Background context explaining how random walks are used to learn node embeddings by exploring paths in the interaction graph. These embeddings can then be used for recommendations.

:p What is a random walk-based approach in GNNs?
??x
A random walk-based approach in GNNs involves generating sequences of nodes (paths) by randomly traversing the user-item interaction graph. The learned node embeddings are then used to make recommendations based on these paths, capturing high-order connections and leveraging graph structure.

```java
// Pseudocode for Random Walk Generation
public class RandomWalkGenerator {
    public List<List<Node>> generateRandomWalks(Node startNode, int walkLength) { ... }
}
```
x??

---

#### MetaPaths in GNNs
Background context explaining the breaking of the assumption that nodes are homogeneous and introducing meta-paths to handle heterogeneous types. This allows for more nuanced learning between different node types.

:p What is a metapath approach in GNNs?
??x
A metapath approach in GNNs breaks the assumption that all nodes are homogeneous, allowing for co-embedding of different node types through defined paths (meta-paths). This approach provides more nuanced learning and can handle complex relationships between heterogeneous node types.

```java
// Pseudocode for Metapath Definition
public class MetaPath {
    public String getPathType() { ... }
    public List<Node> getNodesOnPath(Node startNode) { ... }
}
```
x??

---

