# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 17)

**Starting Chapter:** BERTopic A Modular Topic Modeling Framework

---

#### BERTopic Overview
Background context explaining that BERTopic is a modular topic modeling framework. It uses clusters of semantically similar texts and combines them with classic bag-of-words techniques to model distributions over words.

:p What is BERTopic, and how does it combine different techniques?
??x
BERTopic is a modular topic modeling framework that leverages clusters of semantically similar documents created through embedding and clustering. It then uses a bag-of-words technique, specifically a class-based TF-IDF approach (c-TF-IDF), to model distributions over words in the corpus's vocabulary.
x??

---

#### First Step in BERTopic Pipeline
Background context explaining that the first step involves creating clusters of semantically similar documents by embedding and clustering.

:p What does the first part of BERTopic's pipeline entail?
??x
The first part of BERTopic’s pipeline is to create clusters of semantically similar documents. This involves:
1. Embedding documents using a Transformer model.
2. Reducing the dimensionality of these embeddings.
3. Clustering the reduced embeddings into groups of semantically similar documents.

This step helps in identifying and grouping texts that are contextually similar, which forms the basis for topic modeling.
x??

---

#### Bag-of-Words Technique
Background context explaining that BERTopic uses a classic bag-of-words technique to count word frequencies at a document level.

:p How does BERTopic use the bag-of-words technique?
??x
BERTopic uses a classic bag-of-words technique to generate term frequency (TF) values, which count the number of times each word appears in a document. This can be used to extract the most frequent words within a single document.
```python
from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer()
X_bow = vectorizer.fit_transform(documents)
```
x??

---

#### c-TF-IDF Calculation
Background context explaining that BERTopic modifies the bag-of-words by calculating word frequencies at the cluster level and applying an IDF weighting.

:p How does BERTopic modify the bag-of-words technique to account for more meaningful words?
??x
BERTopic addresses the limitations of the document-level bag-of-words representation by:
1. Calculating word frequency within entire clusters rather than individual documents.
2. Applying a class-based variant of TF-IDF (c-TF-IDF) that emphasizes words significant to specific clusters while downweighting common stop words.

The c-TF-IDF is calculated as follows:
$$\text{c-TF-IDF} = \text{c-TF} \times \text{IDF}$$

Where $\text{c-TF}$ is the frequency of a word within its cluster, and $\text{IDF}$ is defined as:
$$\text{IDF}(w) = \log\left(\frac{\text{total number of clusters}}{1 + \text{frequency of } w \text{ across all clusters}}\right)$$

This weighting helps in identifying more meaningful topics by giving higher scores to words that are important within specific clusters.
x??

---

#### Pipeline Summary
Background context summarizing the two main steps in BERTopic: creating semantically similar document clusters and generating word distributions using c-TF-IDF.

:p What are the two key steps involved in BERTopic?
??x
The two key steps in BERTopic are:
1. Creating clusters of semantically similar documents.
2. Generating a distribution over words in the corpus's vocabulary by leveraging class-based TF-IDF (c-TF-IDF).

This framework allows for topic modeling that accounts for both semantic similarity and meaningful word distributions.
x??

---

#### BERTopic Pipeline Overview
BERTopic's pipeline consists of two main steps: clustering and topic representation. Clustering groups similar documents, while representing topics involves calculating the weight of terms within each cluster to generate representative keywords.

The full pipeline involves:
1. **Clustering**: Grouping semantically similar documents.
2. **Topic Representation**: Assigning a specific ranking of vocabulary items to each cluster, highlighting key terms that represent the topic.

:p What is BERTopic's two-step process?
??x
BERTopic processes text data in two steps: first by clustering similar documents and then representing topics through term weighting within those clusters.
x??

---

#### Clustering in BERTopic
Clustering groups semantically similar documents. The goal here is to identify cohesive sets of documents based on their content, which form the basis for generating meaningful topics.

:p What is the main purpose of the clustering step in BERTopic?
??x
The main purpose of the clustering step in BERTopic is to group semantically similar documents together, forming clusters that can be further analyzed and represented as topics.
x??

---

#### Topic Representation
After clustering, topic representation calculates the weight of terms within each cluster. This step assigns a specific ranking to vocabulary items, making it possible to generate keywords that represent the themes in each cluster.

:p How does BERTopic determine which words are representative of a topic?
??x
BERTopic determines which words are representative by calculating their weights within each cluster and assigning them a specific ranking. The higher a word's weight, the more representative it is of that topic.
x??

---

#### Modularity in BERTopic
Modularity allows for flexibility in the pipeline components. Each part of the pipeline—embedding model, clustering algorithm, and topic representation technique—is replaceable with another similar algorithm.

:p What does modularity allow in BERTopic?
??x
Modularity in BERTopic allows you to use different algorithms or models within each step of the pipeline (e.g., embedding, clustering, and topic representation), making it highly adaptable to various use cases.
x??

---

#### Algorithmic Variants Supported by BERTopic
BERTopic supports a wide range of algorithmic variants for generating topics. This flexibility includes guided, semi-supervised, hierarchical, dynamic, multimodal, multi-aspect, online/incremental, and zero-shot topic modeling.

:p What types of topic modeling does BERTopic support?
??x
BERTopic supports various types of topic modeling such as guided, semi-supervised, hierarchical, dynamic, multimodal, multi-aspect, online/incremental, and zero-shot models.
x??

---

#### Implementation Example with BERTopic
To run BERTopic on the ArXiv dataset, you can use previously defined models for embedding, UMAP, and HDBSCAN. The following code demonstrates how to initialize and fit a model:

:p How do you initialize and fit a BERTopic model?
??x
```python
from bertopic import BERTopic

# Initialize the model with predefined models
topic_model = BERTopic(
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    verbose=True
)

# Fit the model to the abstracts and embeddings
topic_model.fit(abstracts, embeddings)
```
x??

---

#### Customization in BERTopic
BERTopic’s modularity allows you to customize various aspects of the pipeline. For example, you can swap the embedding model from `sentence-transformers` with any other technique or replace clustering algorithms like HDBSCAN with k-means.

:p How can you customize BERTopic?
??x
BERTopic can be customized by replacing different components in its pipeline such as embedding models (`embedding_model`), UMAP models (`umap_model`), and clustering algorithms (`hdbscan_model`). This modularity enables flexibility in adapting the model to specific use cases.
x??

---

#### Overview of get_topic_info() Method
The `get_topic_info()` method is a utility function that provides a quick overview of the topics discovered using BERTopic. It returns information on each topic, including its name and the most representative keywords. The output helps in understanding the main themes or subjects present in the dataset.
:p What does the get_topic_info() method do?
??x
The `get_topic_info()` method provides a summary of the topics found by BERTopic, listing them with their names and top keywords. This summary is useful for quickly grasping the major themes within the text data.
```python
topic_model.get_topic_info()
```
x??

---

#### Topic Representation in get_topic_info()
In the `get_topic_info()` output, each topic is represented by a name that includes its four most representative keywords concatenated with underscores. This helps in understanding what each topic primarily discusses.
:p How are topics named and identified using get_topic_info()?
??x
Topics in the `get_topic_info()` output are named based on their top four keywords, which are concatenated using an underscore. For example, a topic might be labeled as "speech_asr_recognition_end" indicating it deals with automatic speech recognition.
```python
topic_model.get_topic_info()
```
x??

---

#### Handling Outliers in Topic Modeling
The first topic in the `get_topic_info()` output is labeled -1 and contains documents that could not be clustered. These are considered outliers, a result of using HDBSCAN which does not force all points to cluster.
:p What is the significance of the -1 topic in get_topic_info()?
??x
The -1 topic represents documents that were not successfully assigned to any other topics due to their unique characteristics or lack of similarity. It serves as an indicator for data points that are outliers according to HDBSCAN clustering.
```python
topic_model.get_topic_info()
```
x??

---

#### Using get_topic() Function to Explore Topics
The `get_topic()` function can be used to inspect individual topics and see which keywords best represent them. By examining the output, you can understand what a specific topic is about based on its top keywords.
:p How do you use the get_topic() function?
??x
To explore an individual topic using BERTopic, the `get_topic()` function is used. It returns the most representative keywords for that particular topic, allowing for detailed examination of the topic's content.
```python
topic_model.get_topic(0)
```
x??

---

#### Finding Topics with find_topics()
The `find_topics()` function allows you to search for specific topics based on a given term. This is useful when you want to focus on certain themes within your dataset.
:p How does the find_topics() method work?
??x
The `find_topics()` method searches through all discovered topics and returns those that are most similar to a provided search term. It helps in quickly identifying relevant topics based on keywords or phrases of interest.
```python
topic_model.find_topics("topic modeling")
```
x??

---

#### Inspecting Topics with find_topics()
Using the `find_topics()` function, you can identify which topics match your search query and verify if BERTopic has correctly identified them by checking their keywords.
:p How do you verify a topic found using find_topics()?
??x
To verify a topic found using `find_topics()`, inspect its top keywords to ensure they align with the expected theme. For example, searching for "topic modeling" would check if the returned topics contain relevant terms like 'topic', 'topics', and 'lda'.
```python
topic_model.get_topic(22)
```
x??

---

#### Visualizing Documents and Topics
To better understand the relationship between documents and topics, BERTopic provides an interactive visualization that shows which documents belong to each topic. This helps in exploring the data more intuitively.
:p How can you visualize the relationships between documents and their associated topics?
??x
Using `visualize_documents()`, you can create an interactive plot showing the document-topic associations. This allows for a visual exploration of how different documents cluster around specific topics, enhancing understanding through visualization.
```python
fig = topic_model.visualize_documents(titles, reduced_embeddings=reduced_embeddings)
```
x??

---

#### Visualizing Bar Charts with Ranked Keywords
Bertopic also offers a bar chart visualization that ranks keywords within each topic. This can help in identifying the most prominent terms contributing to a particular topic.
:p How do you create a bar chart of ranked keywords for topics?
??x
To visualize the ranked keywords for each topic, use `visualize_barchart()`. This function generates a bar chart where the height of each bar represents the importance of the keyword in that topic.
```python
topic_model.visualize_barchart()
```
x??

---

#### Visualizing Topic Relationships
In addition to individual topic visualizations, BERTopic also provides tools to explore relationships between different topics. This can help in understanding how related or distant topics are from each other.
:p How do you visualize the relationships between different topics?
??x
To visualize relationships between topics, use `visualize_relationships()`. This function creates a graph where nodes represent topics and edges indicate their similarity based on shared documents or keywords.
```python
topic_model.visualize_relationships()
```
x??

---

#### Reranking Initial Topic Representations

Background context: The initial topic representations generated by BERTopic using c-TF-IDF might not be optimally descriptive or semantically meaningful. To improve these representations, a reranking step can be applied using various techniques. This process is akin to refining the results of an initial search query using more sophisticated methods.

:p How does BERTopic enhance topic representation after the initial c-TF-IDF distribution?

??x
BERTopic enhances topic representation by applying a reranking technique that uses advanced models, such as embedding-based methods, to refine and improve the keywords associated with each topic. This process helps in making the topics more semantically meaningful while retaining computational efficiency.

For example, consider a scenario where BERTopic generates an initial set of top 5 words for a topic: "speech | asr | recognition | end | acoustic." A reranker like KeyBERTInspired could refine this list by removing redundant terms and adding more contextually relevant ones. The updated representation might be: "speech | encoder | phonetic | language | trans..."

The key idea is to leverage the fast c-TF-IDF method for initial topic extraction, then use slower but more powerful models (like KeyBERTInspired) to fine-tune these topics.

```python
from bertopic.representation import KeyBERTInspired

# Example of using KeyBERTInspired reranker
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### KeyBERTInspired Representation Model

Background context: KeyBERTInspired is a method used in BERTopic to refine the initial topic representations generated by c-TF-IDF. It works by calculating the similarity between document embeddings and those of the corresponding topic, effectively reranking the words for each topic.

:p How does KeyBERTInspired work to improve topic representation?

??x
KeyBERTInspired works by first extracting the most representative documents per topic using c-TF-IDF. Then, it calculates the average embedding of these documents and compares them with the embeddings of candidate keywords. The goal is to rerank the initial set of top words for each topic based on their similarity to the document embeddings.

For example, if a topic's representation initially includes "speech | asr | recognition | end | acoustic," KeyBERTInspired might refine it by considering more contextually relevant terms like "encoder" and "phonetic."

```python
from bertopic.representation import KeyBERTInspired

# Example of using KeyBERTInspired reranker
representation_model = KeyBERTInspired()
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### Maximal Marginal Relevance (MMR)

Background context: MMR is an algorithm used to select a set of keywords that are diverse from each other but still relevant to the documents they represent. This technique helps in reducing redundancy and improving the semantic quality of topic representations.

:p What is the purpose of using maximal marginal relevance (MMR) in BERTopic?

??x
The purpose of using MMR in BERTopic is to refine the keywords associated with each topic by ensuring that selected words are diverse yet still contextually relevant. This process helps in removing redundant terms and focusing on those that provide the most unique information about the topic.

For instance, if a topic's representation initially includes "speech | asr | recognition | end | acoustic," MMR might refine this list to include more specific or varied terms like "encoder" and "phonetic."

The algorithm works by embedding candidate keywords and iteratively selecting the next best keyword that maximizes relevance while minimizing redundancy.

```python
# Example of using MMR for representation
from bertopic.representation import MMR

representation_model = MMR(diversity=0.5)  # Set diversity parameter
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### Visualizing Topic Hierarchies and Heatmaps

Background context: BERTopic provides tools to visualize the hierarchical structure of topics using heatmaps and hierarchies. These visualizations help in understanding the relationships between different topics.

:p How can BERTopic be used to visualize potential topic hierarchies?

??x
BERTopic can be used to visualize potential topic hierarchies by leveraging its `.visualize_hierarchy()` method, which shows a dendrogram representing the hierarchical clustering of topics. Additionally, the `.visualize_heatmap(n_clusters=30)` method allows for visualizing the top 30 clusters and their relationships.

For example:

```python
# Visualizing hierarchy
topic_model.visualize_hierarchy()

# Visualizing heatmap with 30 clusters
topic_model.visualize_heatmap(n_clusters=30)
```

These visualizations provide insights into how topics are grouped and related to each other, making it easier to understand the structure of the topic model.

x??

---

#### Saving Original Topic Representations

Background context: It is often useful to save the original topic representations before applying any reranking models. This allows for easy comparison between the initial and refined topic representations.

:p Why is it important to save the original topic representations in BERTopic?

??x
Saving the original topic representations is important because it enables a clear comparison between the initial c-TF-IDF based topic representations and those refined by applying reranking models. This process helps in evaluating the effectiveness of different representation models on the topics.

For example, if you save the original topic representations using:

```python
original_topics = deepcopy(topic_model.topic_representations_)
```

You can then compare them with the updated representations after applying a reranking model like KeyBERTInspired or MMR. This comparison helps in assessing whether the refinement has improved the quality of the topics.

x??

---

#### Importing Maximal Marginal Relevance for Topic Representation
To enhance our topic representations, we can use the `MaximalMarginalRelevance` method from BERTopic. This technique helps to make the topics more diverse and representative.

:p How does the `MaximalMarginalRelevance` method work in improving topic representation?

??x
The Maximal Marginal Relevance (MMR) method aims to balance information gain and diversity among topics. It selects words that maximize relevance to a topic while minimizing redundancy within the same topic, thus providing a more nuanced and diverse set of keywords for each topic.

This can be achieved by using the `MaximalMarginalRelevance` class from BERTopic as follows:

```python
from bertopic.representation import MaximalMarginalRelevance

# Initialize the MMR model with a specified diversity parameter
representation_model = MaximalMarginalRelevance(diversity=0.2)

# Update the topic representations using the abstracts and the MMR model
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### Using Text Generation for Topic Labeling
We can use generative models to create short labels for topics based on the keywords and documents related to each topic. This approach is more efficient than generating or reranking keywords for millions of documents.

:p How does BERTopic integrate a text generation model for topic labeling?

??x
BERTopic integrates a text generation model by creating prompts that include both relevant documents and keywords as inputs. The generative model then generates a short label based on this input, providing more concise and meaningful labels than traditional keyword extraction methods.

For example, using the Flan-T5 model:

```python
from transformers import pipeline
from bertopic.representation import TextGeneration

# Define the prompt for generating topic labels
prompt = """I have a topic that contains the following documents: [DOCUMENTS] 
The topic is described by the following keywords: '[KEYWORDS]'. 
Based on the information above, what is this topic about?"""

# Initialize the text generation model from Flan-T5
generator = pipeline("text2text-generation", model="google/flan-t5-small")

# Define the TextGeneration representation model with the generator and prompt
representation_model = TextGeneration(generator=generator, 
                                      prompt=prompt,
                                      doc_length=50, 
                                      tokenizer="whitespace")

# Update topics using the text generation model
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### Using OpenAI's GPT-3.5 for Topic Labeling
OpenAI’s GPT-3.5 can generate more informative and precise labels compared to other models due to its larger size and advanced language capabilities.

:p How does BERTopic use the OpenAI API to generate topic labels?

??x
BERTopic uses the OpenAI API by defining a prompt that includes both relevant documents and keywords as inputs. The GPT-3.5 model then generates a short, descriptive label based on this input.

Here is an example of how to do it:

```python
import openai
from bertopic.representation import OpenAI

# Define the prompt for generating topic labels
prompt = """I have a topic that contains the following documents: [DOCUMENTS] 
The topic is described by the following keywords: [KEYWORDS]. 
Based on the information above, extract a short topic label in the format:
topic: <short topic label>"""

# Initialize the OpenAI client with your API key
client = openai.OpenAI(api_key="YOUR_KEY_HERE")

# Define the OpenAI representation model with the GPT-3.5 model
representation_model = OpenAI(
    client=client, 
    model="gpt-3.5-turbo", 
    exponential_backoff=True,
    chat=True,
    prompt=prompt
)

# Update topics using the text generation model from OpenAI
topic_model.update_topics(abstracts, representation_model=representation_model)
```

x??

---

#### Visualizing Topics with DataMapPlot Package
After generating new topic labels, we can visualize the documents and their associated topics using the `visualize_document_datamap` method. This visualization helps to understand which documents belong to each topic.

:p How does BERTopic use the `datamapplot` package for visualizing topics?

??x
BERTopic uses the `datamapplot` package to visualize the relationship between documents and their associated topics. The `visualize_document_datamap` method generates a plot that shows which documents belong to each topic, making it easier to understand the distribution of content across different topics.

Here is an example:

```python
# Visualize the document-datamap
fig = topic_model.visualize_document_datamap(
    titles,
    topics=list(range(20)),
    reduced_embeddings=reduced_embeddings,
    width=1200,
    label_font_size=11,
    label_wrap_width=20,
    use_medoids=True
)
```

The `visualize_document_datamap` method provides a clear and interactive visualization, making it easier to identify which documents belong to each topic.

x??

---

