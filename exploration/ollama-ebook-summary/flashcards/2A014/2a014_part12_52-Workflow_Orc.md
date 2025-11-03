# Flashcards: 2A014 (Part 12)

**Starting Chapter:** 52-Workflow Orchestration

---

#### Microservice vs. Monolithic Architectures
Background context: In web applications, there are two common architectural styles for deploying models—microservices and monoliths. Microservices involve breaking down a system into smaller, independent services that communicate via APIs, while monoliths keep all the necessary logic in one application.
:p What is the main difference between microservice and monolithic architectures?
??x
Microservices involve decomposing an application into small, autonomous services that communicate over well-defined APIs. Monolithic architectures encapsulate all components within a single application. Microservices offer flexibility and scalability but require managing multiple services, whereas monoliths simplify deployment and maintenance.
x??

---

#### Application Size and Memory Requirements
Background context: Depending on the complexity of your models and their inference requirements, you need to consider the size and memory needs of your application.
:p How do large datasets at inference time impact your application?
??x
Large datasets at inference time can significantly increase memory requirements. You must ensure that your application has sufficient memory to handle these datasets efficiently without causing performance issues or crashes. Techniques like caching, offloading data processing tasks to disk, or using streaming can help manage this.
x??

---

#### Access and Dependency Management
Background context: Your model might depend on external services such as feature stores or databases. Managing these dependencies is crucial for the proper functioning of your application.
:p What should you consider when deciding how to access your models?
??x
When accessing your models, consider whether they will be built in memory within the application or accessed via API calls. Tight coupling with resources like feature stores can simplify development but might require careful management and coordination between services.
x??

---

#### Single Node vs. Cluster Deployment
Background context: For certain model types, parallelizing inference steps may be necessary to achieve performance benefits. This often involves deploying your models on a cluster rather than a single node.
:p What are the considerations for deploying models on a single node versus a cluster?
??x
Deploying models on a single node is simpler and requires less configuration but can become a bottleneck when handling large datasets or high traffic. On the other hand, deploying on a cluster allows for better scalability and parallel processing capabilities, though it involves more complex setup and management.
x??

---

#### Replication for Availability and Performance
Background context: To ensure availability and performance, you might need to run multiple instances of your service simultaneously. This is achieved through replication, where each instance operates independently but can be managed via container orchestration tools like Kubernetes.
:p What role does horizontal scaling play in ensuring the robustness of your application?
??x
Horizontal scaling involves running multiple copies of the same service simultaneously to distribute load and improve availability. Each replica operates independently and can fail without affecting others, ensuring that the overall system remains functional. Tools like Kubernetes help manage these services by coordinating them through strategies such as rolling updates and load balancing.
x??

---

#### Exposing Relevant APIs
Background context: Clear API definitions are essential for integrating your model with other parts of the system. These APIs should specify the expected input and output formats, allowing other applications to call them seamlessly.
:p What is the importance of defining clear schemas in your API?
??x
Defining clear schemas ensures that all components of your system can communicate effectively by agreeing on data formats and structures. This reduces errors and enhances maintainability. For example, a well-defined schema might specify JSON formats for inputs and outputs, with clear documentation on required fields.
x??

---

#### Spinning Up a Model Service with FastAPI
Background context: FastAPI is a framework that simplifies building web APIs to serve machine learning models. It allows for rapid prototyping and development of robust services.
:p How can you use FastAPI to turn a trained torch model into a service?
??x
To turn a trained torch model into a service with FastAPI, you first initialize the FastAPI application, then use Weights & Biases to load your model from an artifact store. You define endpoints that receive user inputs and return predictions.
```python
from fastapi import FastAPI
import wandb, torch

app = FastAPI()
run = wandb.init(project="Prod_model", job_type="inference")
model_dir = run.use_artifact('bryan-wandb/recsys-torch/model:latest', type='model').download()
model = torch.load(model_dir)
model.eval()

@app.get("/recommendations/{user_id}")
def make_recs_for_user(user_id: int):
    endpoint_name = 'make_recs_for_user_v0'
    logger.info({"type": "recommendation_request", f"arguments": {"user_id": user_id}, 
                 f"response": None, f"endpoint_name": {endpoint_name}})
    recommendation = model.eval(user_id)
    logger.log({"type": "model_inference", f"arguments": {"user_id": user_id},
                f"response": recommendation, f"endpoint_name": {endpoint_name}})
    return {"user_id": user_id, "endpoint_name": endpoint_name, "recommendation": recommendation}
```
x??

---

#### Workflow Orchestration
Background context: Beyond the model service itself, you need to orchestrate workflows involving data collection, preprocessing, and inference. This includes containerization, scheduling, and CI/CD pipelines.
:p What are the key components of workflow orchestration?
??x
Workflow orchestration involves several key components:
- Containerization: Ensuring consistent environments across different services using tools like Docker.
- Scheduling: Managing job execution using cron or triggers to coordinate tasks in an ML pipeline.
- CI/CD: Automating tests, validation, and deployment processes to streamline development and production cycles.
x??

---

#### Schemas and Priors
Schemas and priors are foundational principles when designing software systems, especially in serving models. They encompass expected behaviors and assumptions that need to be validated or checked during system operation. These expectations can range from simple data type validations to more complex distributional properties of latent spaces.

:p What is the importance of schemas and priors in software design?
??x
Schemas and priors are essential because they ensure that components within a system adhere to expected behaviors, which can help prevent runtime errors and improve overall robustness. For instance, assuming a user_id will always be correctly typed and existing in the latent space helps maintain predictability and reliability.

```java
public class Example {
    public Vector lookupUserRepresentation(String userId) throws InvalidUserIdException {
        if (userId == null || !isValidUserId(userId)) {
            throw new InvalidUserIdException("Invalid or missing userId");
        }
        // logic to fetch representation
    }

    private boolean isValidUserId(String userId) {
        return true;  // assume validation function exists here
    }
}
```
x??

---

#### Latent Space Representations and Distributions
Latent spaces are used in various applications, such as user-item recommendations, where each entity (e.g., users or items) is represented by a vector. These vectors have specific domains that must be maintained to ensure the integrity of the model.

:p How can you ensure the correct domain for representation vectors in latent space?
??x
Ensuring the appropriate domain for representation vectors involves estimating these distributions as part of the training procedure and storing them for inference. This can be achieved using techniques like calculating KL divergence to monitor how well the embeddings fit within their expected range.

```java
public class LatentSpaceMonitor {
    public double calculateKLDivergence(Vector expected, Vector actual) {
        // logic to compute Kullback-Leibler Divergence between two vectors
        return 0.0;  // placeholder for calculation result
    }
}
```
x??

---

#### Cold-Start Problem in Latent Spaces
Cold-start problems occur when a user or item does not have a sufficient representation in the latent space, leading to a need for alternative prediction pipelines.

:p How can you handle cold-start problems in recommendation systems?
??x
Handling cold-start problems involves transitioning to different prediction methods such as user-feature-based recommendations, explore-exploit strategies, or hardcoded recommendations when the primary latent space representations are insufficient. This requires understanding fallback mechanisms and implementing them gracefully.

```java
public class RecommendationSystem {
    public Item recommendForUser(String userId) {
        try {
            // attempt to use main model
        } catch (ColdStartException e) {
            return fallbackRecommendation(userId);
        }
        return recommendedItem;
    }

    private Item fallbackRecommendation(String userId) {
        // logic for alternative recommendation methods
        return new Item();
    }
}
```
x??

---

#### Integration Testing and Entanglement Issues
Integration testing in complex systems can reveal issues where multiple components interact unexpectedly, leading to what is sometimes referred to as "entanglement" problems.

:p What are some strategies to address entanglement issues during integration testing?
??x
Strategies to address entanglement issues include allowing callbacks from filtering steps to retrieval and building user distribution estimates. The first approach involves dynamic adjustments in the retrieval process, while the second involves pre-calculating appropriate k-values based on user behavior.

```java
public class IntegrationTest {
    public void testRecommendationSystem() {
        // call representation space with k=20
        List<Item> items = retrieveKItems(userId, 20);
        List<Item> filteredItems = filter(items, userId);

        if (filteredItems.isEmpty()) {
            items = retrieveKItems(userId, 50);  // dynamic adjustment
            filteredItems = filter(items, userId);
        }
    }

    private List<Item> retrieveKItems(String userId, int k) {
        // logic to retrieve k items
        return new ArrayList<>();
    }

    private List<Item> filter(List<Item> items, String userId) {
        // logic to apply filters
        return new ArrayList<>();
    }
}
```
x??

---

#### Over-Retrieval in Recommendation Systems
Over-retrieval is a technique used to mitigate issues arising from conflicting requirements by retrieving more potential recommendations than strictly necessary.

:p Why is over-retrieval important in recommendation systems?
??x
Over-retrieval is crucial because it allows downstream rules or personalization algorithms to filter out irrelevant items, thereby ensuring that only suitable recommendations are shown. This prevents the system from failing when faced with conflicting requirements or poor personalization scores.

```java
public class RecommendationSystem {
    public List<Item> retrieveAndFilter(String userId) {
        List<Item> retrievedItems = retrieveMoreThanNeeded(userId);
        return applyFilters(retrievedItems, userId);
    }

    private List<Item> retrieveMoreThanNeeded(String userId) {
        // logic to retrieve more items than needed
        return new ArrayList<>();
    }

    private List<Item> applyFilters(List<Item> items, String userId) {
        // filter out irrelevant items based on user preferences
        return new ArrayList<>();
    }
}
```
x??

---

#### Observability in Distributed Systems
Observability tools help understand the state and behavior of software systems by tracing requests through multiple services. Spans and traces are key concepts in this context.

:p What is the difference between spans and traces in observability?
??x
Spans refer to the time delays or durations of individual service calls, while traces represent the sequence of these calls across services. Together, they provide a detailed picture of how requests flow through distributed systems.

```java
public class TraceExample {
    public void requestHandling(String userId) {
        Span span1 = startSpan("User Lookup");
        String userRepresentation = lookupUser(userId);
        
        Span span2 = startSpan("Item Retrieval");
        List<Item> items = retrieveItemsForUser(userRepresentation);
        
        // process and return results
    }

    private Span startSpan(String name) {
        // logic to create a span with the given name
        return new Span();
    }
}
```
x??

---

#### Observability and Traces
Background context: Observability is a crucial aspect of system monitoring that enables you to see traces, spans, and logs. This allows for appropriate diagnosis of system behavior by understanding how different parts of your service interact over time. For instance, when using a callback from the filter step to get more neighbors from the collector, observability helps trace multiple calls and identify performance bottlenecks.

:p What is the purpose of observing traces and spans in the context of service interactions?
??x
Observing traces and spans provides insight into how different parts of your service interact over time. By visualizing these interactions, you can diagnose issues such as slow responses due to excessive or redundant calls between services. This helps in identifying areas where performance improvements are needed.
x??

---

#### Timeouts and Fallbacks
Background context: Timeouts are hard restrictions implemented to prevent a process from running indefinitely and causing poor user experience. In the context of recommendation systems, timeouts ensure that responses do not take too long. A fallback mechanism is essential when a timeout occurs, as it provides an alternative action or response to minimize delays.

:p What is the role of timeouts in preventing bad user experiences?
??x
Timeouts are implemented to limit how long a process can run before being forcibly stopped to prevent poor user experience. For example, if a recommendation system takes too long to generate a response, a timeout ensures that the request does not hang indefinitely. The fallback mechanism, such as using precomputed recommendations (MPIR), helps maintain service availability even when the primary process times out.
x??

---

#### Evaluation in Production
Background context: Evaluating models in production involves extending model validation techniques beyond training to assess real-world performance. This includes looking at how the model performs on live data and measuring its impact on business KPIs such as revenue.

:p What does evaluation in production entail?
??x
Evaluation in production extends model validation techniques by assessing the model's real-world performance using live data. It involves monitoring metrics like recommendation distributions, affinity scores, candidate numbers, and other ranking scores to ensure that the model behaves as expected in a live environment.
x??

---

#### Slow Feedback Mechanisms
Background context: Recommendation systems often have delayed feedback loops where it takes weeks or even longer to see the impact of recommendations on business KPIs such as revenue. This delay makes it challenging to measure causality and understand the effectiveness of new models.

:p What is slow feedback, and why is it a challenge in recommendation systems?
??x
Slow feedback refers to the long loop from making a recommendation to seeing its impact on metrics like revenue. Because this process can take weeks or longer, it is difficult to establish clear causal relationships between model performance and business KPIs. This delay poses challenges for running experiments and rolling out new models.
x??

---

#### Model Metrics in Production
Background context: Key production metrics help track the model's performance during inference. These include distributions of recommendation categories, affinity scores, candidate numbers, and other ranking scores. Comparing these with precomputed distributions using KL divergence can provide insights into the model's behavior.

:p What are some key metrics to track for a model in production?
??x
Key metrics to track for a model in production include:
- Distribution of recommendations across categorical features
- Distribution of affinity scores
- Number of candidates
- Distribution of other ranking scores

These metrics help monitor the model’s performance and behavior during inference. Comparing these distributions with precomputed ones using techniques like KL divergence can reveal unexpected patterns or issues.
x??

---

