# Flashcards: 2A014 (Part 9)

**Starting Chapter:** 42-PySpark

---

#### Hydration Process in Data Pipelines
Background context: In data processing for machine learning, getting data into the pipeline is often referred to as "hydration." This term originates from the ML and data fields where water-themed naming conventions are common. The process involves collecting logs or raw event data and transforming it into a structured format suitable for model training.
:p What does the term "hydration" refer to in the context of data processing?
??x
The process of getting data into the pipeline, often involving the transformation of log files or event streams into a structured form that can be used for training machine learning models. This is akin to preparing ingredients before cooking a meal, where raw materials are transformed and organized.
x??

---

#### Using PySpark for Data Processing
Background context: PySpark provides a SQL API that allows you to write queries against large datasets in a distributed manner using Spark. It helps in transforming log files into structured data suitable for model training while leveraging the power of distributed computing.
:p How does PySpark facilitate data processing?
??x
PySpark utilizes its SQL API to allow writing queries similar to those used with traditional databases, but these operations are performed on large-scale datasets across multiple nodes. The lazy evaluation nature and distributed architecture enable efficient handling of big data without loading everything into memory at once.

Example code snippet:
```python
user_item_view_counts_qry = """\
SELECT
  page_views.authenticated_user_id
  , page_views.page_url_path
  , COUNT(DISTINCT page_views.page_view_id) AS count_views
FROM prod.page_views
JOIN prod.dim_users
ON page_views.authenticated_user_id = dim_users.authenticated_user_id
WHERE DATE page_views.view_tstamp >= '2017-01-01'
AND dim_users.country_code = 'US'
GROUP BY
  page_views.authenticated_user_id
  , page_views.page_url_path
ORDER BY 3, page_views.authenticated_user_id
"""
user_item_view_counts_sdf = spark.sql(user_item_view_counts_qry)
```
x??

---

#### Lazy Evaluation in Spark
Background context: One of the key features of PySpark is lazy evaluation. This means that queries are not executed immediately when defined but are staged for execution until they are needed by a downstream operation. This approach minimizes unnecessary computation and optimizes resource usage.
:p What is lazy evaluation in the context of Spark?
??x
Lazy evaluation in Spark refers to the mechanism where operations on data are not performed immediately upon definition, instead being deferred until an action (downstream processing) triggers their execution. This allows for efficient use of resources by avoiding unnecessary computations.

Example:
```python
# Lazy evaluation example
user_item_view_counts_qry = """\
SELECT
  page_views.authenticated_user_id
  , page_views.page_url_path
  , COUNT(DISTINCT page_views.page_view_id) AS count_views
FROM prod.page_views
JOIN prod.dim_users
ON page_views.authenticated_user_id = dim_users.authenticated_user_id
WHERE DATE page_views.view_tstamp >= '2017-01-01'
AND dim_users.country_code = 'US'
GROUP BY
  page_views.authenticated_user_id
  , page_views.page_url_path
ORDER BY 3, page_views.authenticated_user_id
"""
# The query is not executed here
user_item_view_counts_sdf = spark.sql(user_item_view_counts_qry)
```
x??

---

#### Distributed Computing in Spark
Background context: Spark is a distributed computing framework that processes data across multiple nodes. The driver program coordinates the execution of tasks on worker nodes, enabling efficient handling of large datasets.
:p How does Spark achieve distributed processing?
??x
Spark achieves distributed processing through its architecture where a driver program coordinates with a cluster manager (like YARN or Mesos) to manage executors running on worker nodes. Data is divided and processed in parallel across these nodes, allowing for scalable and efficient handling of big data.

Example diagram:
```
+---------------------------------------+
|          Driver Program               |
+---------------------------------------+
              |
              v
+---------------------------------------+
|  Cluster Manager (YARN/Mesos)         |
+---------------------------------------+
              |
              v
+-------------------+       +-------------------+
| Worker Node 1     | ...   | Worker Node N    |
+-------------------+       +-------------------+
```
x??

---

#### Offline Collector for Recommendation Systems
Background context: In the context of recommendation systems, an offline collector is responsible for gathering and transforming data to train models. PySpark can be used to create datasets from log files that are suitable for training these models.
:p What role does PySpark play in building offline collectors?
??x
PySpark plays a crucial role in building offline collectors by enabling the transformation of raw event logs (like page view logs) into structured data that can be used to train machine learning models. This involves querying and aggregating log data, applying filters, and generating useful features.

Example query:
```python
item_popularity_qry = """\
SELECT
  page_views.page_url_path
  , COUNT(DISTINCT page_views.authenticated_user_id) AS count_viewers
FROM prod.page_views
JOIN prod.dim_users
ON page_views.authenticated_user_id = dim_users.authenticated_user_id
WHERE DATE page_views.view_tstamp >= '2017-01-01'
AND dim_users.country_code = 'US'
GROUP BY
  page_views.page_url_path
ORDER BY 2
"""
item_view_counts_sdf = spark.sql(item_popularity_qry)
```
x??

---

#### Parsing and Aggregating Log Data
Background context explaining how parsing log data for aggregation can be done efficiently using PySpark. It highlights the difference between doing this with simple SQL-like operations versus leveraging PySpark's distributed computing capabilities.

:p How does PySpark help in processing large-scale log data?
??x
PySpark allows for efficient and scalable processing of large datasets by distributing computations across multiple worker nodes. This enables handling massive amounts of log data without manual partitioning or specifying which nodes should perform specific tasks, as these are handled automatically by the framework.

```python
from pyspark.sql import SparkSession

# Initialize a Spark session
spark = SparkSession.builder \
    .appName("LogDataProcessing") \
    .getOrCreate()

# Example SQL query to parse log data into DataFrame
log_data_qry = """
SELECT
  user_id,
  item_id,
  timestamp,
  action
FROM prod.log_data
"""
parsed_log_data_sdf = spark.sql(log_data_qry)

# Further processing with PySpark API
aggregated_logs = parsed_log_data_sdf \
    .groupBy('user_id', 'item_id') \
    .agg({'timestamp': 'count'}) \
    .withColumnRenamed('count(timestamp)', 'action_count')
```
x??

---

#### Storing Aggregated Data for Item Popularity
Background context explaining the benefits of storing aggregated item popularity data in a database or memory to avoid parsing during query execution.

:p Why might we store aggregated item popularity data instead of parsing log data every time it’s needed?
??x
Storing pre-aggregated item popularity data in a database or memory (e.g., as an in-memory cache) reduces the need for real-time parsing and aggregation, thereby improving response times. This approach is particularly useful when dealing with frequent queries that don’t require up-to-the-minute accuracy.

```python
# Example of storing aggregated item popularity in an in-memory data structure
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("ItemPopularityStorage") \
    .getOrCreate()

# Aggregate logs and store top-N items in memory
aggregated_logs = parsed_log_data_sdf.groupBy('item_id').agg({'timestamp': 'count'})
top_items = aggregated_logs.orderBy(col('count').desc()).limit(100)

# Store in memory for quick access
top_items.cache()
```
x??

---

#### Spark’s Horizontal Scalability
Background context explaining how Spark scales horizontally by adding more worker nodes, making it suitable for processing large datasets.

:p How does Spark achieve horizontal scalability?
??x
Spark achieves horizontal scalability by allowing the addition of more worker nodes to a cluster. This means that as the size of the dataset grows, you can simply add more machines to handle the increased load without modifying your application code significantly. This is in contrast to vertical scaling where you might need to upgrade hardware on existing machines.

```python
# Example configuration for adding workers
spark = SparkSession.builder \
    .appName("HorizontalScalingExample") \
    .master("local[*]")  # Use all available cores locally, or specify a cluster URL
    .config("spark.executor.instances", "4")  # Add 4 more executors (workers)
    .getOrCreate()
```
x??

---

#### PySpark’s Expressiveness and Flexibility
Background context explaining the power of PySpark by comparing it to pandas and SQL, highlighting its ability to perform complex operations with ease.

:p How does PySpark provide an advantage over traditional SQL when dealing with large datasets?
??x
PySpark offers a combination of Python's flexibility and the distributed computing capabilities of Spark. It allows users to write code that looks similar to pandas or SQL but can be executed in a distributed manner across multiple nodes, making it ideal for large-scale data processing.

```python
# Example combining PySpark with Pandas-like operations
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("PySparkExample") \
    .getOrCreate()

df = spark.read.csv("path/to/data")

# Convert to pandas DataFrame (optional)
pdf = df.toPandas()

# Perform complex transformations using PySpark API
aggregated_df = df.groupBy('column1', 'column2').agg({'value': 'sum'})
```
x??

---

#### User Similarity in Collaborative Filtering
Background context explaining the importance of user similarity for recommending items to users based on similar behavior.

:p What is the concept of user similarity and how can it be computed using PySpark?
??x
User similarity measures how alike two users are based on their interaction patterns. In collaborative filtering, if users A and B have similar ratings for common items, then item recommendations from one user can be recommended to another. This can be computed by comparing the deviations of their ratings from their average.

```python
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("UserSimilarity") \
    .getOrCreate()

# Load and preprocess data
ratings_df = spark.read.csv("path/to/ratings")

# Compute current rating as most recent for each user-item pair
windows = Window.partitionBy(['book_id', 'user_id']).orderBy(col('rating_tstamp').desc())
ratings_df = ratings_df.withColumn("current_rating", first("rating_value").over(windows))

# Calculate average and deviation from mean
ratings_df = ratings_df.withColumn("user_avg_rating", avg("current_rating").over(Window.partitionBy('user_id')))
ratings_df = ratings_df.withColumn("deviation_from_mean", col("current_rating") - col("user_avg_rating"))

# Self-join to compute similarity
similarities = (
    ratings_df.alias("left_ratings")
    .join(ratings_df.alias("right_ratings"),
          (col("left_ratings.book_id") == col("right_ratings.book_id"))
           & (col("left_ratings.user_id") != col("right_ratings.user_id")),
          "inner"
         )
    .select(
        col("left_ratings.book_id").alias("book_id"),
        col("left_ratings.user_id").alias("user_id_1"),
        col("right_ratings.user_id").alias("user_id_2"),
        col("left_ratings.deviation_from_mean").alias("dev_1"),
        col("right_ratings.deviation_from_mean").alias("dev_2")
    )
    .withColumn("similarity", (col("dev_1") * col("dev_2") / ((sqrt(col("dev_1")) * sqrt(col("dev_2")))))
)
```
x??

---

#### Generating Affinity Matrix
Background context explaining the calculation of affinity between users based on their similarity scores and item ratings.

:p How can we generate an affinity matrix for items using user similarity scores?
??x
To generate an affinity matrix, you first calculate the similarities between users. Then, using these similarities, compute the weighted average of item ratings to estimate the appropriateness of each item for a given user.

```python
# Example formula and logic for generating affinity matrix
from pyspark.sql import SparkSession

spark = SparkSession.builder \
    .appName("AffinityMatrix") \
    .getOrCreate()

# Precomputed user similarities and items per user
user_similarities = spark.read.csv("path/to/user_similarities")
items_per_user = spark.read.csv("path/to/items_per_user")

# Calculate affinity for each item by weighted sum of similar users' ratings
affinity_matrix = (
    user_similarities.crossJoin(items_per_user)
    .withColumn("weight", col("similarity"))
    .groupBy("user_id", "item_id")
    .agg(sum(col("weight") * col("rating")).alias("weighted_rating"))
    .withColumnRenamed("user_id", "A")
    .withColumnRenamed("item_id", "i")
)
```
x??

---

#### Mini-Batched Gradient Descent

Background context: During training via gradient descent, we make a forward pass of our training sample through our model to yield a prediction. We then compute the error and the appropriate gradient backward through the model to update parameters. However, as datasets scale, computing gradients over all data at once becomes infeasible.

:p What is mini-batched gradient descent?
??x
Mini-batched gradient descent is an approach where we compute gradients of the loss function for a subset (mini-batch) of the dataset rather than the entire dataset. This reduces memory requirements and increases computational efficiency.
??x

#### Stochastic Gradient Descent (SGD)

Background context: SGD is the simplest paradigm for mini-batched gradient descent, computing these gradients and parameter updates one sample at a time. While computationally more intensive per iteration, it can help in avoiding local minima due to its stochastic nature.

:p How does Stochastic Gradient Descent (SGD) differ from traditional batch gradient descent?
??x
In contrast to batch gradient descent which processes the entire dataset for each update, SGD processes one sample at a time. This makes it computationally less intensive per iteration but requires more iterations to converge.
??x

---

#### Jacobians in Mathematics

Background context: The mathematical notion of a Jacobian is an organizational tool for vector derivatives with relevant indexes. For functions of several variables, the Jacobian can be written as a row vector of first derivatives.

:p What is a Jacobian?
??x
A Jacobian is a matrix (or a vector for scalar functions) that contains all the first-order partial derivatives of a vector-valued function or a multivariable function. It generalizes the gradient to vector-valued functions.
??x

---

#### DataLoaders in PyTorch

Background context: DataLoaders provide an efficient way to handle large datasets by batching and shuffling data. They are crucial for training deep learning models on massive datasets, reducing memory overhead and improving computational efficiency.

:p What is the primary purpose of a DataLoader?
??x
The primary purpose of a DataLoader is to facilitate mini-batch access from large datasets efficiently, managing memory usage and providing parallelized batch generation.
??x

---

#### Code Example for DataLoaders

Background context: The following code snippet demonstrates how to use PyTorch's DataLoader API.

:p Provide an example of creating a DataLoader in PyTorch.
??x
```python
params = {
    'batch_size': 32,
    'shuffle': True,
    'num_workers': 4
}

training_generator = torch.utils.data.DataLoader(training_set, params)
validation_generator = torch.utils.data.DataLoader(validation_set, params)

# Training loop example
for epoch in range(max_epochs):
    for local_batch, local_labels in training_generator:
        # Model computations
```
??x

---

