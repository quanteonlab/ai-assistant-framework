# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 20)

**Starting Chapter:** In-Context Learning Providing Examples

---

#### In-Context Learning Overview
In-context learning involves providing examples to an LLM to guide its behavior. This approach can be more effective than describing tasks in words, as it directly shows the desired outcome.

Background context: This method is useful when you want to teach an LLM a specific task or style without explicitly detailing all aspects of the task. By showing examples, the model learns through demonstration rather than instruction.
:p How does in-context learning differ from traditional prompting?
??x
In-context learning differs from traditional prompting by leveraging provided examples instead of just describing the desired outcome. This can make the learning process more intuitive and effective for the LLM.

Example code to illustrate:
```python
def example_prompt():
    # Traditional prompting
    prompt = "Explain how in-context learning works."
    
    # In-context learning with an example
    example_input = "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
    example_output = "The musician played a beautiful melody on his Gigamuru during the festival."

    return [example_input, example_output]
```
x??

#### Zero-shot Prompting
Zero-shot prompting refers to providing no examples or guidance to an LLM when giving it a task.

Background context: In zero-shot prompting, you directly ask the model to perform a task without showing any examples. This method tests the model's ability to generalize from its training data.
:p What is zero-shot prompting?
??x
Zero-shot prompting involves asking an LLM to complete a task or generate content without providing any examples of how it should be done.

Example code:
```python
def zero_shot_prompt():
    # Zero-shot prompt example
    prompt = "Explain the concept of in-context learning."
    
    return prompt
```
x??

#### One-shot Prompting
One-shot prompting involves showing a single example to an LLM when giving it a task.

Background context: This method provides a single instance to the model, allowing it to learn from that one example before performing the requested task.
:p What is one-shot prompting?
??x
One-shot prompting involves providing a single example to an LLM and then asking it to perform a similar task based on that example.

Example code:
```python
def one_shot_prompt():
    # One-shot prompt example
    input_example = "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:"
    output_example = "The musician played a beautiful melody on his Gigamuru during the festival."

    return [input_example, output_example]
```
x??

#### Few-shot Prompting
Few-shot prompting involves showing multiple examples to an LLM when giving it a task.

Background context: This method provides several examples, allowing the model to learn from these cases before performing the requested task.
:p What is few-shot prompting?
??x
Few-shot prompting involves providing two or more examples to an LLM and then asking it to perform a similar task based on those examples.

Example code:
```python
def few_shot_prompt():
    # Few-shot prompt example
    inputs = [
        "A 'Gigamuru' is a type of Japanese musical instrument. An example of a sentence that uses the word Gigamuru is:",
        "The artist mastered playing the Gigamuru in just one week."
    ]
    outputs = [
        "The musician played a beautiful melody on his Gigamuru during the festival.",
        "After practice, the student could play complex rhythms on his Gigamuru."
    ]

    return [inputs, outputs]
```
x??

--- 

Each flashcard covers a different aspect of in-context learning and provides context, code examples, and explanations.

#### One-Shot Prompting Overview
One-shot prompting is a technique where you provide a single example to guide the model, allowing it to generate responses that align with the provided context. It's useful when you want to give the AI a clear direction without overwhelming it with too many examples.

:p What is one-shot prompting and how does it work?
??x
One-shot prompting involves providing a single example to the model so that it can understand the desired output format or style. This method helps in guiding the AI to produce responses that are coherent and relevant based on the given context. For instance, if you want the assistant to respond about a specific instrument, you might provide an example like: "I have a Gigamuru that my uncle gave me as a gift."

Example code:
```python
one_shot_prompt = [
    {"role": "assistant", "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."}
]
```
x??

---

#### Chaining Prompts Example
Chaining prompts involves breaking down complex tasks into smaller, manageable parts by using the output of one prompt as input for another. This method allows the model to focus on each part more effectively.

:p How can you chain prompts to generate a product name, slogan, and sales pitch?
??x
You break down the task into three steps: creating a name and slogan first, then using that along with other features to create the slogan, and finally using all details to craft the sales pitch. This allows the model to handle each part more effectively.

Example code:
```python
# Step 1: Create product name and slogan
product_prompt = [
    {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
]

outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]

print(product_description)

# Step 2: Use the product description to create a sales pitch
sales_prompt = [
    {"role": "user", "content": f"Based on {product_description}, write a sales pitch for the chatbot."}
]

outputs = pipe(sales_prompt)
sales_pitch = outputs[0]["generated_text"]

print(sales_pitch)
```
x??

---

#### Context and Differentiation
Understanding context is crucial in prompt engineering. Each piece of information or example provided should be distinct to avoid confusion between the user's request and the assistant's response.

:p How does providing multiple examples help differentiate the user from the assistant?
??x
Providing multiple examples helps clarify the context by demonstrating how different interactions are handled. For instance, using one example where a Gigamuru is mentioned helps the model understand that subsequent sentences are part of the conversation with the user. Without proper differentiation, there could be ambiguity in distinguishing between user inputs and assistant responses.

Example code:
```python
one_shot_prompt = [
    {"role": "assistant", "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."},
    {"role": "user", "content": "To 'screeg' something is to swing a sword at it."}
]
```
x??

---

#### Prompt Engineering Techniques
Prompt engineering involves using various techniques like one-shot and chaining prompts to guide the AI model effectively. These methods help in producing coherent and contextually relevant outputs.

:p What are some key techniques used in prompt engineering?
??x
Some key techniques include:
- One-shot prompting: Providing a single example to set the context.
- Chaining prompts: Breaking down complex tasks into smaller, sequential steps.
These techniques help guide the model's output, making it more aligned with user expectations.

Example code:
```python
# One-shot prompt
one_shot_prompt = [
    {"role": "assistant", "content": "I have a Gigamuru that my uncle gave me as a gift. I love to play it at home."},
    {"role": "user", "content": "To 'screeg' something is to swing a sword at it."}
]

# Chaining prompts
product_prompt = [
    {"role": "user", "content": "Create a name and slogan for a chatbot that leverages LLMs."}
]
sales_prompt = [
    {"role": "user", "content": f"Based on {product_description}, write a sales pitch for the chatbot."}
]

# Output from model
outputs = pipe(one_shot_prompt)
print(outputs[0]["generated_text"])

outputs = pipe(product_prompt)
product_description = outputs[0]["generated_text"]

sales_pitch = pipe(sales_prompt)
print(sales_pitch["generated_text"])
```
x??

---

#### Prompt Chaining for Sales Pitch Generation
Background context: The process of using a language model (LLM) to generate a sales pitch by providing it with a product description and slogan. This is done in two steps, where the first step generates a name and slogan for the product, and the second step uses this output to create a more detailed sales pitch.
:p How does prompt chaining help in generating a sales pitch?
??x
Prompt chaining helps by breaking down the task into smaller, manageable parts. The initial step might involve generating a catchy name or slogan based on the product description, which can then be used as input for creating a detailed and compelling sales pitch. This approach ensures that each part of the output is coherent and builds upon the previous one.
x??

---

#### Detailed Sales Pitch Generation
Background context: After generating a name and slogan, an LLM can create a more elaborate sales pitch by utilizing the generated outputs as inputs. The process involves creating two separate prompts—one for generating a short phrase (name/slogan) and another for crafting a longer, detailed sales pitch.
:p What is the advantage of using prompt chaining for generating a detailed sales pitch?
??x
The advantage lies in the flexibility and complexity that can be achieved by dividing the task into smaller steps. The first step generates concise yet impactful outputs like names or slogans, which serve as strong foundations. These are then used to generate longer, more detailed content such as full-length sales pitches. This method allows for better control over the output's quality and ensures that each part is coherent with the previous one.
x??

---

#### Prompt Chaining Use Cases
Background context: The technique of prompt chaining can be applied in various scenarios beyond just sales pitches. It includes validating responses, creating multiple prompts concurrently, or breaking down complex tasks into simpler components to generate comprehensive outputs.
:p What are some practical use cases for prompt chaining?
??x
Some practical use cases include:
- **Response validation**: Double-checking previously generated outputs by asking the LLM to review and confirm them.
- **Parallel prompts**: Creating multiple prompts simultaneously, then merging or refining their outputs.
- **Writing stories**: Using the LLM to write detailed components of a story, such as summaries, character development, and plot beats.
x??

---

#### Chain-of-Thought Reasoning in Generative Models
Background context: A method called "chain-of-thought" (CoT) has been introduced to enhance the reasoning capabilities of generative models. It involves providing examples or instructions that guide the model through a series of logical steps before generating its response.
:p How does chain-of-thought prompting work?
??x
Chain-of-thought prompting works by including step-by-step reasoning in the prompt, which guides the model to think logically and sequentially before generating an answer. This approach helps in solving complex problems that require multi-step thinking, such as mathematical questions or logical puzzles.

Example:
```json
cot_prompt = [
    {"role": "user", "content" : "Roger has 5 tennis balls. He buys 2 more cans of tennis balls. Each can has 3 tennis balls. How many tennis balls does he have now?" },
    {"role": "assistant" , "content" : "Roger started with 5 balls. 2 cans of 3 tennis balls each is 6 tennis balls. 5 + 6 = 11. The answer is 11." },
    {"role": "user", "content" : "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have?" }
]
```

Output example:
```json
{
    "generated_text": "The cafeteria started with 23 apples. They used 20 apples, so they had 23 - 20 = 3 apples left. Then they bought 6 more apples, so they now have 3 + 6 = 9 apples. The answer is 9."
}
```
x??

---

#### Zero-Shot Chain-of-Thought
Background context: An extension of chain-of-thought prompting, zero-shot CoT does not require prior examples but instead asks the model to provide reasoning directly in its response.
:p How does zero-shot chain-of-thought differ from traditional chain-of-thought?
??x
Zero-shot chain-of-thought differs by requiring the LLM to engage in reasoning without being explicitly provided with an example sequence. It uses prompts like "Let's think step-by-step" to initiate a structured thought process within the model.

Example:
```json
zeroshot_cot_prompt = [
    {"role": "user", "content" : "The cafeteria had 23 apples. If they used 20 to make lunch and bought 6 more, how many apples do they have? Let's think step-by-step." }
]
```

Output example:
```json
{
    "generated_text": "The cafeteria started with 23 apples. They used 20 apples, so they had 23 - 20 = 3 apples left. Then they bought 6 more apples, so they now have 3 + 6 = 9 apples. The answer is 9."
}
```
x??

---

#### System 1 vs. System 2 Thinking
Background context: In human reasoning, two types of thinking are often distinguished—System 1, which is fast and automatic, and System 2, which is slow but logical. Generative models can mimic System 1 by automatically generating outputs without self-reflection, while System 2 involves conscious and reflective processes.
:p What are the characteristics of System 1 and System 2 thinking?
??x
System 1 thinking is characterized as fast, automatic, and intuitive, similar to how generative models operate. It requires no explicit reasoning steps and generates responses quickly.

System 2 thinking, in contrast, is slower, logical, and involves conscious reflection. This type of thinking allows for more deliberate and thoughtful analysis before generating a response.

Example:
- System 1: "I know the answer without really thinking about it."
- System 2: "Let's break down this problem step-by-step to ensure we get the right answer."

By encouraging models to engage in reflective reasoning, they can produce outputs that more closely mimic the thoughtful processes of human System 2 thinking.
x??

---

