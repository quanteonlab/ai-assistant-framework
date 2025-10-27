# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 27)

**Starting Chapter:** Transformers for Vision

---

#### Dense Retrieval Overview
Dense retrieval relies on the similarity of text embeddings. This method embeds a search query and retrieves documents with the nearest embeddings to the query's embedding.
:p What is dense retrieval?
??x
Dense retrieval involves using text embeddings to find similar documents by calculating the distance between embeddings. It leverages deep learning models like Transformers to create vector representations for both queries and documents, then finds the closest matches based on these vectors.
```java
// Pseudocode example
EmbeddingModel model = new EmbeddingModel();
Document[] documents = model.getNearestDocuments(queryEmbedding);
```
x??

---

#### RAG Evaluation Metrics
The evaluation of RAG systems requires multiple axes including fluency, perceived utility, citation recall, and citation precision. These metrics are often evaluated via human evaluations or by using LLM-as-a-judge approaches.
:p What are the key metrics for evaluating RAG systems?
??x
Key metrics for evaluating RAG systems include:
- Fluency: Assessing if the generated text is fluent and cohesive.
- Perceived utility: Evaluating whether the answer is helpful and informative.
- Citation recall: Measuring the proportion of statements supported by citations in retrieved documents.
- Citation precision: Checking how often the citations actually support their associated statements.

Additionally, Ragas can be used to evaluate these metrics using LLMs as judges for automated scoring.
x??

---

#### RAG Systems Overview
RAG systems use a generative LLM at the end of the pipeline to formulate answers based on retrieved documents while citing sources. This approach combines retrieval and generation, enhancing search capabilities by leveraging both types of models.
:p What is RAG in the context of search engines?
??x
RAG (Retrieval-Augmented Generation) systems integrate a generative LLM with a retrieval system. They retrieve relevant documents first and then generate answers based on these documents, ensuring that the answers are consistent with the provided context and include citations for sources.
```java
// Pseudocode example
SearchEngine engine = new RAGEngine();
Document[] retrievedDocs = engine.retrieveDocuments(query);
String answer = engine.generateAnswer(retrievedDocs);
```
x??

---

#### Mean Average Precision (MAP)
Mean average precision (MAP) is a metric used to evaluate the quality of search results. It calculates the average relevance score across all queries in a test suite.
:p What is mean average precision and how is it calculated?
??x
Mean average precision (MAP) measures the quality of search results by averaging the precision at relevant documents for each query in a test suite. The formula can be expressed as:

\[ \text{MAP} = \frac{\sum_{i=1}^{N} P@k_i}{N} \]

Where \( N \) is the number of queries, and \( k_i \) represents the position at which the first relevant document appears for each query.

```java
// Pseudocode example
double calculateMAP(List<QueryResult> results) {
    double totalPrecision = 0.0;
    int nQueries = results.size();
    
    for (QueryResult result : results) {
        int k = -1;
        boolean firstRelevantDocumentFound = false;
        
        for (int i = 0; i < result.retrievedDocuments.length; i++) {
            if (result.isDocumentRelevant(i)) {
                k = i + 1;
                totalPrecision += 1.0 / k;
                firstRelevantDocumentFound = true;
                break;
            }
        }
        
        if (!firstRelevantDocumentFound) {
            totalPrecision += 1.0 / (k + 1);
        }
    }
    
    return totalPrecision / nQueries;
}
```
x??

---

#### Multimodal Large Language Models
Multimodal LLMs are models capable of handling different types of data, such as text and images. This capability can significantly enhance their applications in solving complex problems that require reasoning across multiple modalities.
:p What is the significance of multimodality in large language models?
??x
Multimodality allows large language models to process and reason about various types of input data, not limited to just text. For example, a model can analyze an image and answer questions based on it. This capability opens up new possibilities for applications like computer vision integrated with natural language processing.
```java
// Pseudocode example
MultimodalModel model = new MultimodalModel();
String answer = model.answerQuestion(image, question);
```
x??

---

#### Vision Transformer (ViT)
Vision Transformers (ViT) are a method of converting images into numerical representations using an adaptation of the original Transformer technique. This allows for tasks like image classification and reasoning.
:p What is the Vision Transformer?
??x
The Vision Transformer (ViT) is a model that uses the architecture of the original Transformer to process visual data, similar to how it processes text. It converts images into patches, which are then embedded as tokens in a sequence. These tokens can be passed through multiple layers of an encoder and decoder, allowing for tasks like image classification.
```java
// Pseudocode example
Image[] images = ...; // Input images
VisionTransformer vit = new VisionTransformer();
PatchTokens tokens = vit.preprocessImages(images);
Sequence sequence = vit.embedPatches(tokens);
ClassificationOutput output = vit.classify(sequence);
```
x??

---

#### Visual Transformer (ViT) Algorithm
Background context: The provided text discusses the main algorithm behind ViT, which is a type of transformer model designed to process image data by breaking images into patches and treating them as tokens. This method allows for leveraging the powerful sequence modeling capabilities of transformers.

:p What is the key idea behind the Visual Transformer (ViT) algorithm?
??x
The key idea behind the Visual Transformer (ViT) algorithm involves transforming image data into a format that can be processed by transformer models, which are originally designed for text. The process includes breaking down images into patches, linearly projecting them, and then passing these patch embeddings through an encoder.

This approach allows for leveraging the powerful sequence modeling capabilities of transformers on image data, effectively treating visual inputs as if they were textual tokens.
??x
---

#### Multimodal Embedding Models Overview
Background context: The text introduces multimodal embedding models that can capture both textual and visual representations. These models create embeddings in a single vector space, enabling comparison across different modalities like text and images.

:p What are multimodal embedding models?
??x
Multimodal embedding models are designed to generate embeddings for multiple modalities (e.g., text, images) within the same vector space, allowing for comparisons between representations from different sources. This capability enables tasks such as finding images based on textual input or identifying documents related to a given image.

For example, using CLIP, you can find images similar to "pictures of a puppy" by searching through the model's embedding space.
??x
---

#### Contrastive Language-Image Pre-training (CLIP)
Background context: The text highlights CLIP as one of the most well-known and currently used multimodal embedding models. It is designed for pretraining on both language and image modalities, allowing it to generate embeddings that can compare meanings across different types of data.

:p What is Contrastive Language-Image Pre-training (CLIP)?
??x
Contrastive Language-Image Pre-training (CLIP) is a multimodal embedding model that preprocesses both text and image data. During training, CLIP learns to map textual and visual inputs into the same vector space such that similar items are close in this space. This capability enables tasks like image retrieval based on textual queries or document similarity analysis.

CLIP is particularly useful for applications where you need to compare and find similarities between images and text.
??x
---

#### CLIP Overview
CLIP (Contrastive Language-Image Pretraining) is a multimodal embedding model that can generate embeddings for both images and texts. These embeddings lie in the same vector space, allowing them to be compared with each other.

:p What are the key features of CLIP?
??x
CLIP has several key features:
1. It generates embeddings for both images and text.
2. The embeddings from these two modalities lie in the same vector space.
3. These embeddings can be used for various tasks such as zero-shot classification, clustering, search, and image generation.

x??

---

#### Zero-Shot Classification
Zero-shot classification is a capability where CLIP compares the embedding of an image with the descriptions of possible classes to determine which class the image most closely resembles.

:p What is zero-shot classification in the context of CLIP?
??x
In the context of CLIP, zero-shot classification involves comparing the embedding of an image with the embeddings of possible text descriptions (classes) to identify which description best matches the image. The model can classify images into categories that it was not explicitly trained on.

x??

---

#### Clustering
Clustering in the context of CLIP involves grouping both images and a collection of keywords together to determine which keywords are most associated with sets of images.

:p How does CLIP perform clustering?
??x
CLIP performs clustering by embedding both images and text (keywords) into the same vector space. Similarity measures such as cosine similarity can then be used to group images that share common characteristics or related keywords.

x??

---

#### Image Search
Image search with CLIP involves searching through billions of images or texts to quickly find content relevant to a given input image or text.

:p How does CLIP support image search?
??x
CLIP supports image search by generating embeddings for both the query image and the database of images. It then uses similarity metrics like cosine similarity to find the most relevant images based on their embedding proximity in the vector space.

x??

---

#### Multimodal Embeddings Generation
Generating multimodal embeddings with CLIP involves creating representations for both images and text, optimizing them through contrastive learning.

:p How does CLIP generate multimodal embeddings?
??x
CLIP generates multimodal embeddings by training an image encoder to embed images and a text encoder to embed texts. During the training process, it compares these pairs of embeddings using cosine similarity. The goal is to maximize the similarity for similar image-caption pairs and minimize it for dissimilar ones.

Example code (pseudocode) to illustrate the concept:
```python
def train_clip(image_data, text_data):
    # Initialize image encoder and text encoder
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder()

    for batch in data_loader:
        images, texts = batch
        
        # Embed both images and texts
        image_embeddings = image_encoder.embed(images)
        text_embeddings = text_encoder.embed(texts)
        
        # Calculate cosine similarity
        similarities = cos_similarity(image_embeddings, text_embeddings)
        
        # Optimize the encoders to increase similarity for similar pairs and decrease for dissimilar ones
        optimize(image_encoder, text_encoder, similarities)

    return image_encoder, text_encoder
```

x??

---

#### Contrastive Learning Process
Contrastive learning is the method used by CLIP during training. It involves comparing embeddings of images and their corresponding texts to maximize similarity for similar pairs and minimize it for dissimilar ones.

:p What is contrastive learning in CLIP?
??x
Contrastive learning in CLIP refers to a training process where the model compares embeddings generated from image-text pairs. The goal is to optimize the encoders so that embeddings of images and their corresponding captions become more similar while embeddings of unrelated images and texts become less similar.

x??

---

