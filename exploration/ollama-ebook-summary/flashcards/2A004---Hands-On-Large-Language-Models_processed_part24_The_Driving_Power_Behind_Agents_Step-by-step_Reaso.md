# Flashcards: 2A004---Hands-On-Large-Language-Models_processed (Part 24)

**Starting Chapter:** The Driving Power Behind Agents Step-by-step Reasoning

---

#### What Are Agents?
Agents are systems that leverage LLMs to determine which actions they should take and in what order. They can utilize tools for tasks beyond the capabilities of LLMs alone.
:p How do agents differ from chains we have seen so far?
??x
Agents are more advanced because they not only understand queries but also decide on tools to use and when, allowing them to create a roadmap to achieve goals. They can interact with the real world through these tools, extending their capabilities significantly.
```
Example:
agent = Agent(model=LLM_model)
agent.run(task="solve math problem", tool=Calculator())
```
x??

---

#### Tools Used by Agents
Agents use external tools in addition to LLMs to perform tasks that the model cannot do on its own. These tools can range from calculators to search engines or weather APIs.
:p Can you give an example of a tool used by an agent?
??x
An example is using a calculator for mathematical problems, or a search engine to find information online. For instance, if an agent needs to know the current price of a MacBook Pro in USD and convert it to EUR, it can use a search engine to find prices and a calculator to perform the conversion.
```
Example:
agent = Agent(model=LLM_model)
agent.run(task="find and convert price", tool=SearchEngine())
agent.run(task="convert currency", tool=Calculator())
```
x??

---

#### The ReAct Framework
ReAct is a framework that combines reasoning and acting to enable LLMs to make decisions based on external tools. It iteratively follows three steps: Thought, Action, Observation.
:p What are the three steps of the ReAct pipeline?
??x
The three steps in the ReAct pipeline are:
1. **Thought**: The agent thinks about what action it should take next and why.
2. **Action**: Based on the thought, an external tool is triggered to perform a specific task.
3. **Observation**: After performing the action, the agent observes the result, which often involves summarizing the output of the external tool.

Example:
```python
agent = Agent(model=LLM_model)
prompt = "Find and convert the price of MacBook Pro from USD to EUR."
thought = agent.reason(prompt)  # Generate a thought about the task.
action = thought.action  # Identify the action, e.g., use a search engine or calculator.
result = action.execute()  # Execute the action using an external tool.
observation = agent.observe(result)  # Summarize and observe the result.
```
x??

---

#### Iterative Process in ReAct
The process of ReAct is iterative. After observing, the agent refines its thought based on the observation, leading to more accurate actions and better outcomes.
:p How does ReAct ensure accuracy in its processes?
??x
ReAct ensures accuracy through an iterative process where after each action, the agent observes the results. Based on this observation, it can refine its reasoning (thought) for future actions, leading to a more accurate and effective sequence of steps.

Example:
```python
while not goal_reached:
    thought = agent.reason(prompt)
    action = thought.action
    result = action.execute()
    observation = agent.observe(result)
    prompt = observation  # Update the prompt based on new information.
```
x??

---

#### Example Scenario: Holiday Shopping
In this scenario, an agent searches for current prices of a MacBook Pro and converts USD to EUR using a calculator.
:p Can you outline the steps an agent would take in the given example?
??x
The agent would follow these steps:
1. **Thought**: The LLM thinks about what it needs to do next—searching for prices online.
2. **Action**: Trigger a search engine to find current prices of MacBook Pro in USD.
3. **Observation**: Observe and summarize the results from the search engine.
4. **Action (if needed)**: If prices are found, use a calculator to convert USD to EUR based on known exchange rates.

Example:
```python
agent = Agent(model=LLM_model)
prompt = "Find and convert the price of MacBook Pro from USD to EUR."
thought1 = agent.reason(prompt)  # Reason about searching for prices.
action1 = thought1.action  # Use a search engine.
result1 = action1.execute()  # Execute the search.
observation1 = agent.observe(result1)  # Observe and summarize results.

if result1.contains_prices():
    thought2 = agent.reason(observation1)  # Reason about converting prices.
    action2 = thought2.action  # Use a calculator for conversion.
    result2 = action2.execute()  # Execute the conversion.
    observation2 = agent.observe(result2)  # Observe and summarize results.
```
x??

#### ReAct Process in LangChain
Background context: The ReAct process in LangChain involves an agent that follows a structured approach of thoughts, actions, and observations to answer questions or perform tasks. This is particularly useful for complex instructions where direct answers are not sufficient.

:p What is the ReAct template used for in this context?
??x
The ReAct template is designed to guide the LLM through a structured process where it first formulates its thoughts (Thought), then decides on actions and their inputs (Action and Action Input), performs these actions, observes the results (Observation), and repeats this cycle until it can provide a final answer. The template ensures that the agent systematically interacts with tools and external systems as needed.

```java
// Example of how the ReAct template might be used in code
public class ReActExample {
    public String processQuestion(String input) {
        // Logic to format the question using the ReAct template
        String formattedQuestion = generateReActTemplate(input);
        return executeAgent(formattedQuestion);
    }

    private String generateReActTemplate(String input) {
        // Template generation logic here
        return "Answer the following questions as best you can. You have access to the following tools: {tools} Use the following format: Question: " + input + " Thought: ... Action: ... Action Input: ... Observation: ... Thought: I now know the final answer Final Answer: ... Begin.";
    }

    private String executeAgent(String formattedQuestion) {
        // Logic to interact with the agent and tools
        return "The current price of a MacBook Pro in USD is $2,249.00. It would cost approximately 1911.65 EUR with an exchange rate of 0.85 EUR for 1 USD.";
    }
}
```
x??

---

#### Tools Used in the Example
Background context: The example uses specific tools such as a DuckDuckGo search engine and a math tool that provides basic calculator functionality. These tools are integrated into the agent to enable it to perform complex tasks.

:p What are the two main tools used in this LangChain example?
??x
The two main tools used in this LangChain example are:
1. A DuckDuckGo search engine, which serves as a general web search tool.
2. An LLM-math tool that provides basic calculator functionality for mathematical operations.

These tools allow the agent to access external information and perform calculations necessary to answer complex questions.

```java
// Example of how the search tool might be used in code
public class ToolExample {
    public String searchWeb(String query) {
        DuckDuckGoSearchResults search = new DuckDuckGoSearchResults();
        return search.run(query);
    }
}

// Example of how the math tool might be used in code
public class MathToolExample {
    public double calculate(double amount, double exchangeRate) {
        LLMMath calculator = new LLMMath(); // Assume this is a mock class for demonstration
        return calculator.calculate(amount * exchangeRate); // Simple multiplication example
    }
}
```
x??

---

#### Agent Creation in LangChain
Background context: The agent creation process involves using the `create_react_agent` function from LangChain, which sets up an agent capable of performing tasks by following a structured ReAct template. This setup includes tools for web search and basic math operations.

:p How is the React agent created in this example?
??x
The React agent is created using the `create_react_agent` function from LangChain. The process involves setting up the LLM, defining the tools available to the agent, and creating a prompt template that guides the agent through its thought-action-observation cycle.

```java
// Example of how the React agent might be created in code
public class AgentCreationExample {
    public AgentExecutor createAgent() {
        // Setup the OpenAI LLM
        ChatOpenAI openaiLLM = new ChatOpenAI("gpt-3.5-turbo", 0);

        // Define available tools
        Tool searchTool = new Tool(
                "duckduck",
                "A web search engine. Use this to as a search engine for general queries.",
                (String query) -> {
                    DuckDuckGoSearchResults search = new DuckDuckGoSearchResults();
                    return search.run(query);
                });

        List<Tool> tools = loadTools(["llm-math"], openaiLLM)
                .add(searchTool);

        // Create the React template
        String reactTemplate = "Answer the following questions as best you can. You have access to the following tools: {tools} Use the following format: Question: the input question you must answer Thought: you should always think about what to do Agents: Creating a System of LLMs | 221 Action: the action to take, should be one of [ {tool_names} ] Action Input: the input to the action Observation: the result of the action ... (this Thought/Action/Action Input/Observation can repeat N times) Thought: I now know the final answer Final Answer: the final answer to the original input question Begin. Question: {input} Thought: {agent_scratchpad}";

        PromptTemplate prompt = new PromptTemplate(reactTemplate, "tools", "tool_names", "input", "agent_scratchpad");

        // Create and return the agent
        AgentExecutor agentExecutor = createReactAgent(openaiLLM, tools, prompt);
        return agentExecutor;
    }
}
```
x??

---

#### Invocation of the React Agent
Background context: Once the agent is set up, it can be invoked to perform tasks by providing input questions. The agent processes these inputs through its structured ReAct template and uses available tools to provide answers.

:p How does one invoke the React agent with a specific question?
??x
To invoke the React agent with a specific question, you need to pass an input dictionary containing the question to the `invoke` method of the agent executor. The agent will then process this input through its ReAct template and use available tools to provide a comprehensive answer.

```java
// Example of how to invoke the agent in code
public class AgentInvocationExample {
    public void testAgent(String question) {
        // Assuming 'agentExecutor' is already set up
        String result = agentExecutor.invoke(question);
        System.out.println(result); // Print the output from the agent
    }
}
```
x??

---

#### Overview of Semantic Search and RAG

Semantic search is a powerful approach that enables searching by meaning, rather than just keyword matching. This technique is widely adopted in industry due to its significant improvements in search quality.

Background context: The concept of semantic search was popularized after the release of BERT in 2018, which has been used by major search engines like Google and Bing for improving their search capabilities significantly.
:p What does semantic search enable?
??x
Semantic search enables searching based on meaning rather than just keywords. This means that even if a user searches for "best pizza place," the system will be able to understand the context and provide relevant results, even if the exact words are not present in the database.

---
#### Dense Retrieval

Dense retrieval systems rely on embeddings to convert both search queries and documents into numerical vectors and then retrieve the nearest neighbors of the query from a large archive of texts. This method is part of semantic search.

Background context: In dense retrieval, the similarity between the embedding vectors of the query and the document (or set of documents) determines their relevance.
:p How does dense retrieval work?
??x
Dense retrieval works by converting both the search query and the documents into embeddings, then finding the nearest neighbors in the embedding space. This is shown in Figure 8-1.

---
#### Reranking

Reranking involves reordering a subset of results based on their relevance to the query after initial dense retrieval. It's another component of semantic search pipelines.

Background context: Rerankers take an additional input, which is a set of results from a previous step in the search pipeline and score them against the query.
:p What does a reranker do?
??x
A reranker takes a subset of search results and reorders them based on their relevance to the query. This often leads to more accurate and relevant results.

---
#### RAG (Retrieval-Augmented Generation)

RAG systems combine the strengths of retrieval and generation models, providing both factual answers and context from external sources. They are particularly useful in reducing hallucinations and increasing factuality.

Background context: RAG leverages embeddings for dense retrieval to find relevant documents and uses a language model to generate an answer based on this information.
:p What is the main benefit of using RAG systems?
??x
The main benefit of using RAG systems is that they can reduce hallucinations, increase factuality, and ground the generation model on specific datasets. This combination allows for more accurate and reliable answers.

---
#### Example of a RAG System

A generative search system uses an LLM to formulate an answer based on retrieved information from various sources.

Background context: In this example, we will explore how an LLM can be used in conjunction with retrieval methods to generate factual and relevant answers.
:p Can you provide an example of how RAG works?
??x
Sure! An RAG system would first use dense retrieval to find relevant documents for a query. Then it would present these documents to the LLM, which generates a response based on the retrieved information.

Example pseudocode:
```pseudocode
function generateAnswer(query):
    # Step 1: Retrieve relevant documents using dense retrieval
    relevantDocuments = denseRetrieval(query)
    
    # Step 2: Use the language model to generate an answer
    answer = LLM.generateAnswer(relevantDocuments, query)
    return answer
```

x??
This pseudocode outlines how a RAG system works. It first retrieves documents using dense retrieval and then uses an LLM to process these documents and generate a relevant answer.

---
#### Comparison of Dense Retrieval and Reranking

Dense retrieval focuses on finding the nearest neighbors in embedding space, while reranking refines the results by reordering them based on relevance scores.

Background context: Dense retrieval is about retrieving similar content, whereas reranking adjusts the order to make sure the most relevant items are at the top.
:p How do dense retrieval and reranking differ?
??x
Dense retrieval focuses on finding the nearest neighbors in embedding space, while reranking involves reordering a set of results based on their relevance scores. Dense retrieval is about retrieving similar content, whereas reranking refines the order to ensure that the most relevant items are at the top.

---
#### Example RAG System Workflow

An agent using an LLM can reason about its thoughts and take actions, such as searching the web or using a calculator, before generating a response.

Background context: Agents leverage LLMs to make decisions based on external information.
:p How does an agent with an LLM work in a RAG system?
??x
An agent using an LLM works by first reasoning about its thoughts and taking actions. It can search the web, use a calculator, or access other tools before generating a response. This workflow is part of the ReAct framework.

Example pseudocode:
```pseudocode
function processQuery(query):
    # Step 1: Reason about what needs to be done
    action = agent.reasonAbout(query)
    
    # Step 2: Take an action, e.g., search the web or use a calculator
    result = takeAction(action)
    
    # Step 3: Generate a response using the LLM
    answer = LLM.generateAnswer(result)
    return answer
```

x??
This pseudocode illustrates how an agent with an LLM works in a RAG system. The agent first reasons about what action to take, performs that action, and then generates a response based on the results.

