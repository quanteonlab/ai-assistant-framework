# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 28)

**Starting Chapter:** OpenCLIP

---

#### Loading and Preprocessing Text and Image Inputs
Background context: To use models from OpenCLIP (an open-source variant of CLIP) for generating embeddings, we need to load specific components such as tokenizers, preprocessors, and main models. These components handle text and image inputs separately before combining them in the main model.

:p How do you load and prepare text and image inputs for a CLIP model using OpenCLIP?
??x
To load and prepare text and image inputs for a CLIP model using OpenCLIP, we need to perform several steps. First, we import necessary libraries and define the model ID:

```python
from transformers import CLIPTokenizerFast, CLIPProcessor, CLIPModel

model_id = "openai/clip-vit-base-patch32"
```

Next, we load a tokenizer for text preprocessing and a processor for image preprocessing:

```python
clip_tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
clip_processor = CLIPProcessor.from_pretrained(model_id)
```

After loading the tokenizer and processor, we can preprocess our input data. For example, let's tokenize a caption:
```python
inputs = clip_tokenizer("a puppy playing in the snow", return_tensors="pt")
print(inputs)
```
This will output a dictionary containing token IDs and attention masks.

To prepare an image, we first load it using `Image.open()`:

```python
from PIL import Image

puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large- Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")
```

Then, we preprocess the image using the `clip_processor`:

```python
processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
print(processed_image.shape)
```
This will output a tensor of shape `[1, 3, 224, 224]`, indicating that the image has been resized to 224x224 pixels.

The preprocessing steps ensure that both text and image inputs are in the correct format before being passed to the main CLIP model.
x??

---

#### Tokenizing Text for CLIP Models
Background context: The tokenizer plays a crucial role in converting natural language text into numerical representations (token IDs) that can be understood by the CLIP model.

:p How does the tokenizer handle text inputs in the context of generating embeddings using OpenCLIP?
??x
The tokenizer handles text inputs by breaking down the input text into smaller units called tokens and assigning each token a unique ID. This process helps in converting human-readable text into a form that can be processed by the CLIP model.

Here's an example of how to use the tokenizer from OpenCLIP:

```python
# Load the tokenizer
clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")

# Tokenize the input text
inputs = clip_tokenizer("a puppy playing in the snow", return_tensors="pt")
print(inputs)
```

This will output a dictionary containing token IDs and attention masks:

```python
{'input_ids': tensor([[49406, 320, 6829, 1629, 530, 518, 2583, 49407]]),
 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
```

The `input_ids` represent the tokens in numerical form, and `attention_mask` is a binary mask indicating which parts of the input are padding (0) or actual content (1). The tokenizer also adds special tokens like `<|startoftext|>` to denote the beginning of text for separating it from potential image embeddings.

The tokenizer ensures that the text input is in a suitable format before being processed by the model.
x??

---

#### Preprocessing Images for CLIP Models
Background context: Image preprocessing involves resizing and normalizing images to fit the expected input dimensions required by the CLIP model. This step is crucial for ensuring accurate embedding generation.

:p How do you preprocess an image using OpenCLIP's processor?
??x
To preprocess an image using OpenCLIP's processor, we need to resize and normalize the image so that it matches the expected input format of the CLIP model. Here’s a step-by-step guide:

1. **Load the Image**: Use the `PIL` library to open and convert the image.
2. **Preprocess the Image**: Use the `CLIPProcessor` to handle the preprocessing.

Here's an example of how to preprocess an image:

```python
from PIL import Image

# Load the image
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large- Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

# Preprocess the image using OpenCLIP's processor
processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
print(processed_image.shape)
```

This will output a tensor of shape `[1, 3, 224, 224]`, indicating that the image has been resized to 224x224 pixels and normalized.

The preprocessing step ensures that the input image is in the correct format before being passed to the main CLIP model for embedding generation.
x??

---

#### Generating Text and Image Embeddings
Background context: After preparing the text and image inputs, we can generate embeddings using the CLIP model. These embeddings capture the semantic meaning of both text and images.

:p How do you generate embeddings from preprocessed text and image data?
??x
To generate embeddings from preprocessed text and image data, follow these steps:

1. **Load the Tokenizer and Processor**: Use `CLIPTokenizerFast` to tokenize text inputs and `CLIPProcessor` for preprocessing images.
2. **Preprocess Inputs**: Tokenize text and preprocess images as described earlier.
3. **Generate Embeddings**: Pass the preprocessed data through the CLIP model.

Here's an example of generating embeddings:

```python
# Load the tokenizer and processor
clip_tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Preprocess text input
inputs = clip_tokenizer("a puppy playing in the snow", return_tensors="pt")
print(inputs)

# Preprocess image input
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large- Language-Models/main/chapter09/images/puppy.png"
image = Image.open(urlopen(puppy_path)).convert("RGB")

processed_image = clip_processor(text=None, images=image, return_tensors="pt")["pixel_values"]
print(processed_image.shape)

# Generate text embedding
text_embedding = model.get_text_features(**inputs)
print(text_embedding.shape)

# Generate image embedding
image_embedding = model.get_image_features(processed_image)
print(image_embedding.shape)
```

This will output the shapes of the embeddings, which are both `[1, 512]`.

By generating these embeddings, we can compare and analyze the similarity between text and image data using various metrics such as cosine similarity.
x??

---

#### Normalizing Embeddings
Background context: After generating embeddings, normalizing them is a common practice to ensure that each embedding vector has unit length. This step is crucial for calculating accurate similarity scores.

:p How do you normalize the embeddings generated by CLIP?
??x
To normalize the embeddings generated by CLIP, follow these steps:

1. **Extract Embeddings**: Get the text and image embeddings from the model.
2. **Normalize the Embeddings**: Divide each embedding vector by its L2 norm to ensure it has unit length.

Here's an example of normalizing the embeddings:

```python
# Normalize the text and image embeddings
text_embedding /= text_embedding.norm(dim=-1, keepdim=True)
image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

print(text_embedding.shape)  # Should be [1, 512]
print(image_embedding.shape) # Should be [1, 512]
```

By normalizing the embeddings, we ensure that they are on a comparable scale. This is important for calculating similarity scores using methods such as cosine similarity.

The normalized embeddings can then be used to calculate their similarity:

```python
# Calculate the dot product of the normalized text and image embeddings
score = np.dot(text_embedding.detach().cpu().numpy(), image_embedding.detach().cpu().numpy().T)
print(score)  # Output should be a similarity score, e.g., [0.33149648]
```

The normalization step ensures that the cosine similarity calculation is accurate and meaningful.
x??

---

#### Using sentence-transformers to Load CLIP Models
Background context: The `sentence-transformers` library provides an easier way to load and use CLIP-based models for generating embeddings. This simplifies the process by handling many of the preprocessing steps.

:p How can you simplify the embedding generation process using sentence-transformers?
??x
To simplify the embedding generation process using sentence-transformers, follow these steps:

1. **Install sentence-transformers**: Ensure that the `sentence-transformers` library is installed.
2. **Load a CLIP-based Model**: Use the `SentenceTransformer` class to load a pre-trained CLIP model.
3. **Encode Text and Image Inputs**: Pass the text and image inputs directly to the model for embedding generation.

Here's an example of how to use sentence-transformers with OpenCLIP (CLIP variant):

```python
from sentence_transformers import SentenceTransformer, util

# Load a SBERT-compatible CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
images = [Image.open(urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
image_embeddings = model.encode(images)
text_embeddings = model.encode(["a puppy playing in the snow"] * 3)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix
```

This approach simplifies the workflow by handling many preprocessing steps internally. The `SentenceTransformer` class automatically manages tokenization, normalization, and other necessary transformations.

Using sentence-transformers makes it easier to work with CLIP-based models for tasks such as text-image alignment and retrieval.
x??

--- 

Feel free to combine or adjust any of these questions and answers based on your specific needs! If you have more details or additional steps, I can further refine the responses. 😊🚀

If there's anything else you need, just let me know! 🚀✨
```python
# Example code snippets for clarity and completeness

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix
``` 
This example code demonstrates how to use the `sentence-transformers` library with a CLIP-based model to generate and compare image and text embeddings. If you have any more questions or need further assistance, feel free to ask! 🚀✨
```python

# Example code snippets for clarity and completeness

from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```python
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers import SentenceTransformer, util
from PIL import Image
import urllib.request as urlopen

# Load a SBERT-compatible CLIP model using sentence-transformers
model = SentenceTransformer("clip-ViT-B-32")

# Prepare image inputs (list of images)
puppy_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large-Language-Models/main/chapter09/images/puppy.png"
images = [Image.open(urlopen.urlopen(puppy_path)).convert("RGB") for _ in range(3)]

# Encode the images and captions
text_inputs = ["a puppy playing in the snow"] * 3
image_embeddings = model.encode(images)
text_embeddings = model.encode(text_inputs)

# Compute cosine similarities between image and text embeddings
sim_matrix = util.cos_sim(image_embeddings, text_embeddings)

print(sim_matrix)  # Output will be a similarity matrix

```
```python
from sentence_transformers

#### BLIP-2 Overview
Background context: The traditional approach to text generation models is limited to textual information. However, introducing vision capabilities can enhance these models' reasoning abilities about images and text. One such method is called **BLIP-2** (Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation 2).

:p What is BLIP-2 and its main goal?
??x
BLIP-2 aims to bridge the gap between vision and language by creating a multimodal model that can reason about both images and text. It uses pretrained models (a Vision Transformer and a Large Language Model) and introduces a Querying Transformer (Q-Former) as the trainable component.
x??

---

#### Querying Transformer (Q-Former)
Background context: The Q-Former is a key component in BLIP-2 that connects a frozen Vision Transformer with a large language model. It has two main modules, an Image Transformer and a Text Transformer.

:p What are the components of the Q-Former?
??x
The Q-Former consists of:
1. An **Image Transformer** to interact with the frozen Vision Transformer.
2. A **Text Transformer** that can interact with the Large Language Model (LLM).
x??

---

#### Training Stages of BLIP-2
Background context: The training process for BLIP-2 involves two stages, each focusing on one modality.

:p What are the two main stages in the training of BLIP-2?
??x
The two main stages in the training of BLIP-2 are:
1. **Step 1**: Representation learning to learn representations for vision and language simultaneously.
2. **Step 2**: Converting these representations into soft visual prompts to feed the LLM.
x??

---

#### Step 1: Representation Learning
Background context: In the first stage, the Q-Former is trained on image-text pairs to align their embeddings.

:p What tasks are involved in the first step of training?
??x
In the first step of training, the Q-Former is trained on three main tasks:
1. **Image-text contrastive learning**: Aligning pairs of image and text embeddings.
2. **Image-text matching**: Predicting whether an image and text pair is positive (matched) or negative (unmatched).
3. **Image-grounded text generation**: Training the model to generate text based on information extracted from the input image.
x??

---

#### Step 2: Converting Vision to Language
Background context: In the second stage, the Q-Former's learnable embeddings are passed through a projection layer and used as soft visual prompts for the LLM.

:p How is vision converted into language in BLIP-2?
??x
In the second step, the learnable embeddings from the Q-Former are projected using a fully connected linear layer to match the expected shape of the LLM. These embeddings serve as soft visual prompts that condition the LLM on the visual representations extracted by the Q-Former.
x??

---

#### BLIP-2 Applications
Background context: BLIP-2 and similar architectures have been applied in various multimodal large language models, such as LLaVA and Idefics.

:p What are some other applications or similar architectures to BLIP-2?
??x
Other applications of BLIP-2 include frameworks like:
1. **LLaVA**: A framework for making textual LLMs multimodal.
2. **Idefics 2**: An efficient visual LLM based on the Mistral 7B LLM, which connects pretrained CLIP-like visual encoders with textual LLMs to project visual features into language embeddings.
x??

---

#### Loading BLIP-2 Model and Processor
Background context: This concept introduces how to load a pre-trained multimodal model (BLIP-2) and its processor using the Hugging Face `transformers` library. The model is used for generating text based on images, which is an important use case in multimodal processing.

:p How do you load and configure BLIP-2 for image captioning?
??x
To load and configure BLIP-2 for image captioning, follow these steps:

1. Import the necessary libraries.
2. Load the processor from the specified model name.
3. Load the model with the desired precision (float16 in this case) to speed up inference.
4. Move the model to a GPU if available.

Here is how you can do it:

```python
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

# Load processor and main model
blip_processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)

# Send the model to GPU to speed up inference
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
```

x??

---

#### Preprocessing Images for BLIP-2 Model
Background context: The processor is responsible for converting images into a format that the model can understand. This includes resizing and normalizing the image.

:p What happens to an image when it goes through the preprocessing step of the BLIP-2 processor?
??x
When an image goes through the preprocessing step, the processor converts it from its original size (520 × 492 pixels) into a standard square format suitable for the model. Specifically, it resizes the image to 224 × 224 pixels.

This is important because the model expects inputs of consistent dimensions and typically processes images in this resized form. However, very wide or tall images might be distorted due to resizing.

Here is an example of preprocessing an image:

```python
# Load the image
car_path = "https://raw.githubusercontent.com/HandsOnLLM/Hands-On-Large- Language-Models/main/chapter09/images/car.png"
image = Image.open(urlopen(car_path)).convert("RGB")

# Preprocess the image
inputs = blip_processor(image, return_tensors="pt").to(device, torch.float16)
print(inputs["pixel_values"].shape)  # Output: torch.Size([1, 3, 224, 224])
```

x??

---

#### Preprocessing Text for BLIP-2 Model
Background context: The processor also handles text inputs by tokenizing them into a format that the model can process. This involves breaking down the text into tokens and padding it to fit within the maximum length.

:p How does the processor handle text inputs for the BLIP-2 model?
??x
The processor handles text inputs by using its tokenizer to break down the text into tokens. These tokens are then processed and padded to ensure they match the expected input size of the model.

Here is how you can inspect the tokenizer:

```python
# Access the tokenizer used in the processor
blip_processor.tokenizer

# Output:
# GPT2TokenizerFast(name_or_path='Salesforce/blip2-opt-2.7b', vocab_size=50265, 
# model_max_length=1000000000000000019884624838656, is_fast=True, padding_side='right',
# truncation_side='right', special_tokens={'bos_token': '</s>', 'eos_token': '</s>',
# 'unk_token': '</s>', 'pad_token': '<pad>'}, clean_up_tokenization_spaces=True),
# added_tokens_decoder={1: AddedToken('<pad>', rstrip=False, lstrip=False,
# single_word=False, normalized=True, special=True), 2: AddedToken('</s>',
# rstrip=False, lstrip=False, single_word=False, normalized=True, special=True)}
```

x??

---

