# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 24)

**Starting Chapter:** Agent Failure Modes and Evaluation

---

#### Planning Failures Overview
Planning failures occur when an agent fails to execute a task correctly, often due to errors in generating or following a plan. Common issues include tool use failure, invalid parameters for valid tools, and incorrect parameter values.

:p What are some common types of planning failures?
??x
Some common types of planning failures include:
1. **Invalid Tool Use**: The agent generates a plan that includes a tool not available in its inventory.
2. **Valid Tool with Invalid Parameters**: A correct tool is called but with incorrectly formatted or invalid parameters.
3. **Valid Tool with Incorrect Parameter Values**: The correct tool and valid parameters are used, but the values provided are incorrect.
4. **Goal Failure**: The agent fails to achieve the intended goal either because the plan does not solve the task correctly or it violates constraints.

For example:
```java
// Invalid Tool Use Example
Tool bing_search = new BingSearch();
Plan plan = new Plan(bing_search); // This will fail if bing_search is not in the inventory

// Valid Tool with Invalid Parameters Example
Tool lbs_to_kg = new LbsToKg();
plan.addStep(lbs_to_kg.call(100, 2)); // lbs_to_kg requires only one parameter but gets two.
```
x??

---

#### Goal Failure Examples
Goal failure happens when the agent fails to complete a task as intended. This could be due to an incorrect plan or not adhering to constraints.

:p Can you provide an example of goal failure?
??x
Sure, consider an example where you ask the agent to plan a two-week trip from San Francisco to Hanoi with a budget of $5,000:
- The agent might plan a trip to Ho Chi Minh City instead.
- Or it could plan a longer trip that exceeds the budget.

```java
// Example Scenario
Plan plan = new Plan();
// Incorrect destination planning
plan.addStep(new Travel("San Francisco", "Ho Chi Minh City")); // Instead of Hanoi

// Budget constraint failure
plan.setBudget(5000);
plan.addExpenses(6000); // Exceeding the budget
```
x??

---

#### Planning Dataset Metrics for Evaluation
To evaluate planning failures, one can create a dataset where each example consists of (task, tool inventory). The agent is asked to generate multiple plans, and metrics are computed based on validity and other factors.

:p How do you measure the effectiveness of an agent's planning using metrics?
??x
Metrics include:
1. **Valid Plans**: Percentage of generated plans that are valid.
2. **Average Valid Plan Generation Attempts**: Average number of plans needed to get a valid one for each task.
3. **Tool Call Validity**: Fraction of tool calls that are valid.

Example pseudocode:
```java
public class PlanningMetrics {
    public double calculateValidPlansRate(List<Plan> plans) {
        int valid = 0;
        for (Plan p : plans) {
            if (isValid(p)) valid++;
        }
        return (double) valid / plans.size();
    }

    private boolean isValid(Plan plan) {
        // Check if all tool calls in the plan are valid
    }
}
```
x??

---

#### Reflection Errors and Time Constraints
Reflection errors occur when an agent believes it has completed a task, but it hasn't. Time constraints can also be overlooked, making tasks less useful over time.

:p What is reflection error in planning?
??x
Reflection error occurs when the agent concludes that a task has been successfully completed even though it has not. For example:
- Assigning 50 people to 30 rooms: The agent assigns only 40 and claims success.
- Inconsistencies between expected outcomes and actual results.

Example:
```java
// Reflection Error Example
Task assignPeople = new Task(50, 30);
Agent agent = new Agent();
Plan plan = agent.generatePlan(assignPeople);

// Agent incorrectly concludes the task is done
if (plan.isComplete()) {
    System.out.println("All people are assigned.");
} else {
    System.out.println("Some people remain unassigned.");
}
```
x??

---

#### Tool Failures Overview
Tool failures occur when a tool returns incorrect outputs, even though it was correctly used. This can happen due to translation errors or the tool itself providing wrong data.

:p What is an example of a tool failure?
??x
An example of a tool failure could be:
- An image captioner generating incorrect descriptions.
- An SQL query generator producing erroneous queries.

Code Example for Tool Failure:
```java
// Tool Failure Example with Image Captioning
ImageCaptioner captioner = new ImageCaptioner();
String caption = captioner.generateCaption(image);
if (!isValidCaption(caption)) {
    System.out.println("The generated caption is incorrect.");
}
```
x??

---

#### Tool Failures and Efficiency
Background context explaining the concept. Tool failures can happen due to a lack of access to necessary tools or when the agent doesn't have proper authorization to use them. Tool-dependent tasks require independent testing, and each tool call should be printed for inspection. Efficient agents not only complete tasks correctly but do so with minimal resources and time.

:p What are some reasons an agent might experience tool failures?
??x
Tool failures can occur if the agent lacks necessary access or permissions to execute certain tools required for completing a task. For example, attempting to retrieve current stock prices without internet connectivity would result in a tool failure.
```python
# Example of checking internet connection before making a tool call
import requests

def fetch_stock_prices():
    try:
        response = requests.get("https://api.example.com/stocks")
        if response.status_code == 200:
            print(response.json())
        else:
            print("Failed to fetch stock prices due to network issues.")
    except Exception as e:
        print(f"An error occurred: {e}")
```
x??

---
#### Evaluating Agent Efficiency
Background context explaining the concept. To evaluate an agent’s efficiency, consider metrics like the number of steps required, cost incurred, and time taken for actions. Comparing these with a baseline helps in understanding how well the agent performs compared to alternatives.

:p How can you measure the efficiency of an AI agent?
??x
Efficiency can be measured by several key metrics:
- **Number of Steps:** Tracking the average number of steps needed to complete tasks.
- **Cost:** Monitoring the cost per task completion, which may involve resources or financial expenditure.
- **Time Taken:** Measuring how long each action typically takes and identifying any particularly time-consuming actions.

For example, you might compare an AI agent’s performance against a human operator:
```python
def evaluate_agent_efficiency(agent_steps, human_steps):
    efficiency_ratio = (human_steps / agent_steps)
    print(f"Efficiency Ratio: {efficiency_ratio:.2f}")
```
x??

---
#### Role of Memory in RAG and Agents
Background context explaining the concept. Memory is crucial for systems like RAG (Retrieval-Augmented Generation) and agents, as it allows models to retain and utilize information over multiple interactions or steps.

:p Why is memory essential for RAG and agent systems?
??x
Memory is vital for RAG and agent systems because:
- **Knowledge Retention:** It enables the model to store and recall large amounts of information.
- **Contextual Understanding:** For multi-step tasks, maintaining context helps in understanding complex scenarios over time.

For example, a memory mechanism could be implemented using a simple database or cache system:
```python
class MemorySystem:
    def __init__(self):
        self.memory = {}

    def add_info(self, key, value):
        self.memory[key] = value

    def get_info(self, key):
        return self.memory.get(key, None)
```
x??

---
#### Internal Knowledge and Short-term Memory in Models
Background context explaining the concept. AI models have internal knowledge from training data and short-term memory for recent interactions, which helps them generate responses more effectively.

:p What are the two main types of memory mechanisms in a model?
??x
AI models typically use two primary memory mechanisms:
- **Internal Knowledge:** Retained through the model’s training process.
- **Short-term Memory (Context):** Stored in the model’s context for recent interactions or messages.

These mechanisms help models generate more informed and contextually relevant responses.
```python
class ModelMemory:
    def __init__(self):
        self.internal_knowledge = "Pre-trained data"
        self.short_term_memory = []

    def add_to_context(self, message):
        self.short_term_memory.append(message)
```
x??

#### Information Storage Mechanisms
Background context explaining the concept of different information storage mechanisms. This includes internal knowledge, short-term memory, and long-term memory.

The model's internal knowledge is essential for tasks that require immediate processing without external retrieval. Short-term memory holds temporary data relevant to ongoing tasks, while long-term memory stores historical data and can be accessed via external systems such as RAG (Retrieval-Augmented Generation).

:p What are the different types of information storage mechanisms used in AI models?
??x
The model's internal knowledge, short-term memory, and long-term memory. Internal knowledge is integrated into the model’s training, while short-term memory holds context-specific data for current tasks, and long-term memory can be augmented with external data sources.

For example:
- **Internal Knowledge**: Essential information that must be learned through training or finetuning.
- **Short-Term Memory**: Information relevant to immediate context or task execution.
- **Long-Term Memory**: External data accessible via retrieval mechanisms like RAG systems, persisting across tasks without model updates.

x??

---
#### Managing Information Overflow
Background on how models manage information that exceeds their current context limits. This involves storing excess information in long-term memory.

During the process of executing a task, an agent can acquire more information than it can handle at once due to its maximum context length. Excess information is stored in external memory systems to avoid overwhelming the model during execution.

:p How does managing information overflow work in AI models?
??x
Excess information that exceeds the model's current context length is stored in long-term memory, allowing the agent to access it when needed without losing essential data during task execution. This ensures efficient use of internal resources and prevents information overload.

For example:
- The model might store a large document or transcript outside its immediate context but can retrieve parts of it as required.
```java
public class Example {
    public void manageContext(String[] input) {
        if (input.length > MAX_CONTEXT_LENGTH) {
            longTermMemory.store(input);
            String relevantData = longTermMemory.retrieveRelevantData();
            process(relevantData);
        } else {
            process(input);
        }
    }

    private void process(String[] data) {
        // Process the immediate context
    }
}
```
x??

---
#### Persisting Information Between Sessions
Background on how long-term memory helps AI models maintain information across different sessions.

Long-term memory allows an AI model to retain important information that persists between sessions, improving personalization and consistency. Without this mechanism, each session would start from scratch, leading to inefficiencies and poor user experience.

:p How does persisting information between sessions work in AI models?
??x
Persisting information between sessions is achieved by storing historical data or conversation history in long-term memory. This allows the model to recall previous interactions, preferences, or context-specific information when needed, enhancing personalization and consistency across multiple sessions.

For example:
- An AI coach can access your previous advice requests to provide more relevant suggestions.
```java
public class AIAssistant {
    private LongTermMemory memory;

    public void getSessionHistory() {
        String history = memory.retrieveSessionHistory();
        System.out.println("Previous session history: " + history);
    }

    public void provideAdvice(String request) {
        String previousPreferences = memory.retrievePreferences();
        // Use previousPreferences to personalize the advice
        System.out.println("Based on your previous preferences, here is some relevant advice.");
    }
}
```
x??

---
#### Boosting Model Consistency
Background on how referencing previous answers can improve model consistency. This involves maintaining a record of past responses for future reference.

By referencing its previous answers, an AI model can ensure consistency in its responses to similar or repeated questions. This helps maintain coherence and reliability in the model's output across different interactions.

:p How does boosting model consistency work?
??x
Boosting model consistency is achieved by storing and referencing past answers. When a model receives a question it has encountered before, it can use its stored response to ensure that future responses are consistent with previous ones, maintaining coherence and reliability in the output.

For example:
- If asked about the rating of a joke twice, remembering the previous rating ensures consistency.
```java
public class ConsistentModel {
    private HashMap<String, Integer> pastResponses;

    public int rateJoke(String joke) {
        if (pastResponses.containsKey(joke)) {
            return pastResponses.get(joke);
        } else {
            // Calculate new rating and store it for future use
            int rating = calculateRating(joke);
            pastResponses.put(joke, rating);
            return rating;
        }
    }

    private int calculateRating(String joke) {
        // Logic to rate the joke
        return 4; // Example rating
    }
}
```
x??

---
#### Maintaining Data Structural Integrity
Background on why structured data is important in unstructured text and how it can be managed.

Text-based models inherently handle unstructured data, which poses challenges for maintaining structural integrity. Structured data can be fed into the model contextually to ensure that the information is properly formatted and understood by the model.

:p How does maintaining data structural integrity work?
??x
Maintaining data structural integrity involves feeding structured data (e.g., tables) in a way that ensures proper formatting and understanding by the text-based model. Although text inputs are inherently unstructured, using structured formats like line-by-line table entries can help the model process information more effectively.

For example:
- Feeding a table one row at a time allows better handling of tabular data.
```java
public class DataHandler {
    public void feedTable(List<String[]> table) {
        for (String[] row : table) {
            context.appendRow(row);
        }
    }

    private StringBuilder context = new StringBuilder();

    public String getContext() {
        return context.toString();
    }
}
```
x??

#### Memory Management Overview
Memory management involves deciding what information should be stored in short-term and long-term memory. It includes adding and deleting data based on limited storage capacity.

:p What is memory management in the context of AI models?
??x
Memory management refers to the process of determining which pieces of information should be kept in short-term and long-term memory. This involves managing the addition and deletion of data, especially when storage space is limited.
x??

---

#### FIFO Strategy
FIFO (First In, First Out) is a simple strategy for memory management where the first piece of data added to short-term memory is the first one moved to external storage.

:p What does the FIFO strategy do in memory management?
??x
The FIFO strategy ensures that older data gets removed from short-term memory before new data is stored. This means that the first piece of information added will be the first to be moved out when space is needed.
x??

---

#### Usage-Based Strategies
Usage-based strategies involve removing less frequently used information from memory, which requires tracking usage frequency.

:p How do usage-based strategies work in managing memory?
??x
Usage-based strategies determine which pieces of data are accessed most often and remove the least frequently used ones to free up space. This approach involves monitoring the usage frequency of each piece of data to decide whether it should be kept or discarded.
x??

---

#### Redundancy Removal Using Summaries
Summarization can help reduce memory footprint by identifying redundant information, which can then be removed.

:p How does summarization help in managing memory?
??x
Summarization helps manage memory by generating a concise representation of the data, thereby reducing redundancy. This summary can then be used to replace the original detailed content, freeing up space.
x??

---

#### Reflection Approach for Memory Management
The reflection approach involves agents reflecting on new information and deciding if it should be added to or merged with existing memory.

:p What is the reflection approach in managing memory?
??x
The reflection approach involves agents periodically reviewing newly generated information. The agent then decides whether this new data should be inserted into long-term memory, merged with existing memories, or replace outdated information that contradicts the new findings.
x??

---

#### Handling Contradictions in Memory Management
Contradictory pieces of information can cause confusion but also provide different perspectives to draw from.

:p How are contradictions handled during memory management?
??x
Handling contradictions involves deciding whether to keep newer information and potentially discarding older data. Some systems might ask AI models to judge which piece of information should be retained, while others may prioritize newer data or allow the system to make such judgments based on context.
x??

---

#### RAG (Retrieval-Augmented Generation)
Background context: Retrieval-Augmented Generation (RAG) is a pattern that addresses the limitations of models by retrieving relevant information from external memory before generating responses. This approach enhances response quality and efficiency, making it particularly useful for tasks requiring extensive background knowledge.
:p What is RAG, and why was it developed?
??x
RAG is a pattern where a model retrieves relevant information from an external source before generating a response. It addresses the context limitation of models by integrating retrieved data to produce more accurate responses. This method enhances efficiency and response quality while potentially reducing costs.

For example, in code copilots or research assistants, RAG can access large datasets or entire repositories to provide detailed and relevant information.
x??

---

#### Term-Based vs. Embedding-Based Retrievers
Background context: Retriever quality is crucial for the success of a RAG system. Term-based retrievers like Elasticsearch and BM25 are simpler to implement but may not always outperform more complex embedding-based methods. Vector search, which powers embedding-based retrieval, is also used in internet applications such as search engines.
:p What types of retrievers exist, and how do they differ?
??x
Term-based retrievers (e.g., Elasticsearch, BM25) are simpler to implement but may not be as effective as embedding-based methods. Embedding-based retrievers use vector search to provide more accurate results by converting text into numerical vectors.

Example code for a simple term-based retrieval using Elasticsearch:
```java
public class TermBasedRetriever {
    private Client client;

    public TermBasedRetriever(Client client) {
        this.client = client;
    }

    public List<String> retrieve(String query) throws IOException {
        SearchRequest request = new SearchRequest("index_name");
        SearchSourceBuilder builder = new SearchSourceBuilder();
        QueryBuilders.matchQuery("field", query);
        builder.query(builder);
        request.source(builder);

        SearchResponse response = client.search(request, RequestOptions.DEFAULT);
        return Arrays.stream(response.getHits().getHits()).map(hit -> hit.getSourceAsString()).collect(Collectors.toList());
    }
}
```
x??

---

#### Agent Pattern
Background context: The agent pattern involves an AI planner analyzing tasks, considering solutions, and selecting the best one. Agents can solve complex tasks through multiple steps, requiring powerful models with planning capabilities and memory systems to track progress.
:p What is an AI-powered agent?
??x
An AI-powered agent is defined by its environment and tools it can access. The agent uses AI as a planner that analyzes given tasks, considers different solutions, and selects the most promising one. A complex task may require multiple steps, necessitating a powerful model capable of planning.

Example pseudocode for an agent solving a task:
```pseudocode
function solveTask(agent, task) {
    // Analyze the task
    plan = agent.analyze(task)
    
    // Consider different solutions and pick the best one
    best_solution = agent.evaluateSolutions(plan.solutions)
    
    // Execute the chosen solution
    result = agent.execute(best_solution)
    
    return result
}
```
x??

---

#### Security Risks in Agents
Background context: As agents become more automated, they face increased security risks. These risks are discussed in detail in Chapter 5 and need to be mitigated with rigorous defensive mechanisms.
:p What security risks do automated agents pose?
??x
Automated agents can expose organizations to significant security risks as their capabilities increase. These risks include unauthorized access, data breaches, and misuse of tools.

Mitigation strategies involve implementing robust defensive mechanisms such as:
- Access control
- Monitoring and auditing
- Secure communication protocols

These measures help ensure that agents operate within secure boundaries.
x??

---

#### Memory Systems for RAG and Agents
Background context: Both RAG and agents manage large amounts of information, often exceeding the model's context length. A memory system is essential to store and use this information effectively.
:p What is a memory system in RAG and agent patterns?
??x
A memory system manages and uses vast amounts of information that exceed the underlying model’s context length. It allows models to retain progress and access historical data, enhancing their ability to solve complex tasks.

Example code for a simple memory system:
```java
public class MemorySystem {
    private Map<String, Object> data;

    public MemorySystem() {
        this.data = new HashMap<>();
    }

    // Store information
    public void store(String key, Object value) {
        data.put(key, value);
    }

    // Retrieve stored information
    public Object retrieve(String key) {
        return data.get(key);
    }
}
```
x??

---

#### RAG and Agents as Prompt-Based Methods
Background context: RAG and agents operate by influencing the model through input prompts without modifying the underlying model. This approach enables many applications but leaves room for further potential enhancements.
:p How do RAG and agents differ from methods that modify the underlying model?
??x
RAG and agents are prompt-based, meaning they influence the model's quality solely through inputs rather than modifying the model itself. This allows them to enable numerous applications effectively.

In contrast, modifying the underlying model can open up more possibilities but requires a deeper integration into the model architecture, potentially affecting its core operations.

Example:
```java
public class PromptBasedModel {
    public String generateResponse(String prompt) {
        // Process input prompt and generate response using RAG or agent pattern
        return "Response";
    }
}
```
x??

