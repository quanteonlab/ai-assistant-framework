# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 18)

**Starting Chapter:** Context Length and Context Efficiency

---

#### Context Length and Model Capabilities
Context length plays a critical role in how much information can be included in a prompt. The model's context limit has expanded rapidly, from 1K for GPT-2 to 2M for Gemini-1.5 Pro within five years.

The first three generations of GPTs have the following context lengths:
- GPT-1: 1K
- GPT-2: 2K
- GPT-3: 4K

A 100K context length can fit a moderate-sized book, and a 2M context length can accommodate approximately 2,000 Wikipedia pages or a complex codebase like PyTorch.

:p What is the relationship between context length and model capabilities?
??x
Context length directly influences how much information a model can process at once. Longer context lengths allow for more detailed and comprehensive prompts, which can be crucial for tasks requiring extended reasoning or large amounts of input data. For example, 2M context length allows handling complex codebases like PyTorch or extensive documents.

The expansion from 1K to 2M context length has been rapid, indicating a race among model providers to increase this limit.
x??

---

#### Context Efficiency and Prompt Design
The effectiveness of different parts of a prompt depends on their position. Models are better at understanding instructions given at the beginning and end of the prompt than in the middle.

A common test for evaluating context efficiency is called the Needle in a Haystack (NIAH) test, which involves inserting random information (the needle) into various locations within a prompt (the haystack) and asking the model to find it.

:p How does the position of instructions in a prompt affect a model's understanding?
??x
Instructions given at the beginning and end of a prompt are more effectively understood by models compared to those placed in the middle. This is due to the limited attention span or capacity of the model, which means it can handle information better if it’s closer to the start or end.

The NIAH test demonstrates this by showing that all tested models perform better when the information (the needle) is near the beginning and end of the prompt.
x??

---

#### Example NIAH Test
In a study by Liu et al. (2023), an example of a NIAH test was used to evaluate how well different parts of a prompt are understood.

The test involves inserting random information into various positions within a prompt and then asking the model to find that specific piece of information.

:p Can you describe the NIAH test method?
??x
In the NIAH test, a random piece of information (the needle) is inserted at different locations in a prompt (the haystack). The objective is to see how well the model can locate and retrieve this information. For instance, Liu et al.’s paper used a randomly generated string as an example.

Here’s an example:
```plaintext
The quick brown fox jumps over the lazy dog.
: The needle : Some random text
```

In this case, the model would be asked to find "Some random text," and it's observed that models perform better when this information is placed closer to the beginning or end of the prompt.
x??

---

#### Privacy Considerations in Testing
When conducting NIAH tests, it’s essential to ensure that the inserted information is private. This step prevents the model from relying on its internal knowledge rather than the context provided.

If the information used for testing is part of the model's training data, the model might answer based on its pre-existing knowledge instead of analyzing the input context.

:p Why is privacy a concern when using NIAH tests?
??x
Privacy is a critical concern because if the inserted information in an NIAH test is already present in the model’s training data, the model will likely rely on this internal knowledge to answer questions. This approach would not accurately reflect how the model processes new, unseen input.

For instance, if you're testing a doctor visit transcript and include private patient details (like their blood type or medication) that are part of the model's training data, the model might simply recall these details from its training rather than analyzing the context provided in the test prompt.
x??

---

#### Write Clear and Explicit Instructions
Background context explaining the importance of clear instructions when communicating with AI models. Ensuring ambiguity is removed helps in getting precise and relevant responses.

:p What should be included to ensure clarity in a prompt?
??x
To ensure clarity in a prompt, you need to explicitly state what you want the model to do without any ambiguity. For example, if you are asking for an essay score, specify whether it should be on a scale from 1 to 5 or 1 to 10 and provide instructions on handling cases where the model is uncertain.

For instance:
```plaintext
Q: Score this essay from 1 to 10 based on its coherence, relevance, and argument strength. If you're not sure about a particular aspect, output "I don't know."
```
x??

---

#### Ask the Model to Adopt a Persona
Explanation of how adopting a persona can influence the model's response by changing its perspective or tone.

:p How does asking for a model to adopt a persona impact its responses?
??x
Asking the model to adopt a specific persona helps it understand the context and generate more appropriate responses. For example, if you ask a model to score an essay about chickens written from the perspective of a first-grade teacher, the model will provide a more positive and educational response compared to its default setting.

For instance:
```plaintext
Q: Adopt the persona of a first-grade teacher and score this essay on chicken from 1 to 5.
I like chickens. Chickens are fluffy and they give tasty eggs.
```
The model might output a higher score because it understands the perspective of teaching young children about the benefits and attributes of chickens.

x??

---

#### Provide Examples
Explanation of how examples can reduce ambiguity in prompts and guide the model towards the desired response.

:p How do examples help in reducing ambiguity in prompts?
??x
Examples provide context to the model, making its responses more aligned with your expectations. For instance, if you are building a bot that interacts with children and wants it to handle questions about fictional characters like Santa Claus or tooth fairies, providing examples can ensure the model responds appropriately.

For example:
```plaintext
Q: Is the tooth fairy real?
A: Of course. Put your tooth under your pillow tonight. The tooth fairy might visit and leave you something.
```

This example nudges the model to acknowledge the existence of fictional characters in children's stories, which can influence how it responds to other questions like whether Santa brings presents.

x??

---

#### Opt for Example Formats with Fewer Tokens
Explanation on optimizing prompts by choosing example formats that use fewer tokens while maintaining performance.

:p Why should you prefer examples with fewer tokens?
??x
Choosing example formats that use fewer tokens is crucial when dealing with token limitations, especially in contexts where input length is a concern. By opting for more concise examples, you can ensure the model generates responses within its token constraints without compromising on quality.

For instance:
```plaintext
Prompt 1: Label the following item as edible or inedible.
Input: chickpea Output: edible
Input: box Output: inedible
Input: pizza Output:38

Prompt 2 (preferred): 
chickpea --> edible
box --> inedible
pizza -->
```

While both prompts have equal performance, the second one uses fewer tokens and is thus more efficient.

x??

---

#### Specifying Output Format for Models

Background context: The model should generate concise outputs to reduce costs and latency. Avoiding preambles and ensuring correct format is essential when downstream applications require specific formats.

:p How do you specify output format requirements?
??x
You can specify output format by giving clear instructions, such as requesting JSON with particular keys or using markers for structured outputs at the end of prompts.

For example:
- If expecting JSON: `{"key1": "value1", "key2": "value2"}`
- For markers, ensure they don't appear in inputs; e.g., use `<!--END-->` as a marker.
x??

---

#### Dealing with Preambles and Unnecessary Text

Background context: Models might begin responses with preambles that are not needed. These can be avoided by explicitly instructing the model to focus on the task at hand.

:p How do you avoid preambles in your prompts?
??x
Specify in the prompt that you don't want preambles or any unnecessary text. For instance, if labeling items as edible or inedible, directly request a concise answer without introductory statements like "Based on the content..."

Example:
Label the following item as edible or inedible.
pineapple pizza --> edible
cardboard --> inedible

Avoid this:
Based on the content of the given item, pineapple pizza --> edible
x??

---

#### Providing Sufficient Context to Models

Background context: Including relevant context can improve model responses and reduce hallucinations by providing necessary information instead of relying on internal knowledge.

:p How do you provide sufficient context to a model?
??x
Include any reference texts or papers that are relevant to the task. For example, if answering questions about a specific paper, include the paper itself in the prompt.

Example:
Context: "The paper discusses the effects of climate change on agriculture."
Question: What are some key findings?

This ensures the model has the necessary context and avoids incorrect assumptions.
x??

---

#### Restricting Model Knowledge to Provided Context

Background context: Limiting a model's knowledge to only the provided context can be useful for roleplaying or scenarios where the model should not access external information.

:p How do you restrict a model’s knowledge to its context?
??x
Use clear instructions such as “answer using only the provided context” and provide examples of questions it shouldn’t answer. For instance:

```markdown
You are a character in the game Skyrim. Use only the information provided here for your responses.
```

This helps, but isn’t always reliable since models might still access pre-trained knowledge.

Example:
Use only the following text to answer: "A dragon named Drakos is seeking revenge."
x??

---

#### Decomposing Complex Tasks into Subtasks

Background context: Breaking complex tasks into simpler subtasks can enhance model performance and make it easier to manage each step independently. This method also facilitates monitoring, debugging, and parallel processing.

:p How do you break down complex tasks into simpler subtasks?
??x
Decompose the task into smaller prompts that handle specific parts of the process. For example, in a customer support chatbot:

1. Intent classification: Identify the intent of the request.
2. Generating response: Based on this intent, instruct the model how to respond.

Example:
Intent Classification Prompt:
```markdown
SYSTEM
You will be provided with customer service queries. Classify each query into a primary category and a secondary category. Provide your output in json format with the keys: primary and secondary.
Primary categories: Billing, Technical Support, Account Management, or General Inquiry.
Technical Support secondary categories: - Troubleshooting - ...
```

Response Generation Prompt:
```markdown
SYSTEM
You will be provided with customer service inquiries that require troubleshooting in a technical support context. Help the user by: - Ask them to check that all cables to/from the router are connected...
```
x??

---

