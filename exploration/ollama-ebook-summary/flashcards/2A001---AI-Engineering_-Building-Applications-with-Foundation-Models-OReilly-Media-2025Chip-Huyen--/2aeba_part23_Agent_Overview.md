# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 23)

**Starting Chapter:** Agent Overview

---

#### Overview of Agents
Background context explaining the concept. The term agent has been used in various engineering contexts, including software agents, intelligent agents, user agents, conversational agents, and reinforcement learning agents. An agent is anything that can perceive its environment through sensors and act upon it through actuators.
:p What defines an agent?
??x
An agent is defined as something that can perceive its environment through sensors and act upon that environment through actuators. This means the agent has a specific environment in which it operates and a set of actions it can perform based on that environment.
x??

---

#### Agent Environment and Actions
Background context explaining how an agent's environment and actions are related. The environment determines what tools an agent can potentially use, while the tool inventory restricts the environment an agent can operate within.
:p How does the environment affect an agent's capabilities?
??x
The environment affects an agent’s capabilities by defining its operational domain and the set of possible actions it can take. For example, a chess-playing agent operates in a chess game environment where only valid chess moves are allowed as actions.
x??

---

#### Tools for Agents
Background context explaining that tools augment an AI agent's ability to perform tasks. Tools include various functions or systems that an agent uses to accomplish its goals.
:p What is the role of tools in an AI agent?
??x
Tools play a crucial role in augmenting an AI agent’s capabilities by providing functionalities that help it execute specific tasks. For example, ChatGPT can search the web, execute Python code, and generate images.
x??

---

#### Example of SWE-agent
Background context explaining the specifics of the SWE-agent, which operates on a computer with terminal access and file system actions. This agent's environment is the computer terminal and file system, with specific actions like navigating repositories, searching files, viewing files, and editing lines.
:p Describe the environment and actions of SWE-agent?
??x
The SWE-agent operates in an environment consisting of a computer terminal and file system, with actions including navigate repo, search files, view files, and edit lines. This setup allows the agent to perform tasks like code navigation and editing.
x??

---

#### RAG System Actions
Background context explaining that a RAG system has multiple actions for query processing, such as response generation, SQL query generation, and execution. The example given is about projecting sales revenue for Fruity Fedora over three months.
:p What are the key actions in a RAG system?
??x
A RAG system performs several key actions: response generation, SQL query generation, and SQL query execution. For instance, to project future sales revenue for Fruity Fedora, the agent would generate an appropriate SQL query to retrieve historical data and then execute that query.
x??

---

#### Agent's Role in Task Completion
Background context explaining how AI agents process information, plan actions, and determine task completion based on inputs and feedback. Agents like RAG systems are designed to handle tasks typically provided by users.
:p How does an agent accomplish a task?
??x
An AI agent processes information it receives, including the task and feedback from the environment, plans a sequence of actions to achieve this task, and determines whether the task has been accomplished. For example, in the RAG system, the agent reasons about how to predict future sales by generating an SQL query to retrieve historical data.
x??

---

#### Agent's Failures
Background context explaining that agents have new modes of failures due to their complex operations involving tools and planning. Evaluating agents is crucial to catch these failures.
:p What are some challenges in evaluating AI agents?
??x
Evaluating AI agents involves catching the new modes of failure that arise from their complex operations, which include the use of various tools and detailed planning processes. Ensuring that an agent can handle unexpected scenarios and provide accurate results is essential.
x??

---

#### Example of RAG System Workflow
Background context explaining a specific scenario where a RAG system uses tabular data to generate responses and execute SQL queries. The example involves generating a query for sales revenue over three months.
:p Describe the workflow of a RAG system with tabular data?
??x
The workflow of a RAG system with tabular data includes steps like reasoning about how to accomplish the task, invoking SQL query generation to retrieve necessary data, executing the generated query, and using the results to make reliable predictions. For instance, to project future sales revenue for Fruity Fedora, the agent would first reason that it needs historical sales numbers from the last five years.
x??

---

#### Compound Mistakes and Higher Stakes
Background context: The passage discusses how agents, when performing tasks involving multiple steps or complex actions, can face significant accuracy drops due to the cumulative effect of individual mistakes. Additionally, having access to powerful tools increases both the potential impact and risk if something goes wrong.

:p How does the increase in the number of steps in a task affect an agent's overall accuracy?
??x
The increase in the number of steps reduces the overall accuracy exponentially. If each step has a 95% success rate, over 10 steps, the cumulative accuracy drops to approximately 60%, and over 100 steps, it decreases to only about 0.6%.

This can be calculated using the formula:
\[ \text{Overall Accuracy} = (\text{Accuracy per Step})^{\text{Number of Steps}} \]

For example, with a step accuracy of 95% (or 0.95):
\[ \text{Overall Accuracy after 10 steps} = 0.95^{10} \approx 0.60 \]
\[ \text{Overall Accuracy after 100 steps} = 0.95^{100} \approx 0.006 \]

x??

---

#### Tool Inventory and Autonomous Agents
Background context: The passage highlights the importance of an agent's tool inventory in determining its capabilities. Tools enable agents to both perceive and act upon their environment, thereby making them more autonomous.

:p How does an increase in the number of tools affect an agent’s performance?
??x
An increase in the number of tools gives an agent more capabilities but also makes it harder to understand and utilize these tools effectively. Experimentation is necessary to find the right set of tools for optimal performance, as discussed in "Tool selection" on page 295.

For example, consider an agent with three tools:
- Tool A: Provides context.
- Tool B: Allows actions that modify data sources (write actions).
- Tool C: Enables perception of the environment (read-only actions).

x??

---

#### Knowledge Augmentation
Background context: Knowledge augmentation involves providing agents with additional context and information to improve their responses. This can include text, image, or SQL retrieval tools.

:p What are some potential external tools that can augment an agent’s knowledge?
??x
Potential external tools for knowledge augmentation include:
- Text retriever: Fetches relevant documents from internal databases.
- Image retriever: Serves images related to the task at hand.
- SQL executor: Runs queries on structured data sources.

Other examples include:
- Internal people search: Helps find specific individuals within an organization.
- Inventory API: Provides status updates for different products.
- Slack retrieval: Accesses chat history from internal messaging platforms.
- Email reader: Enables interaction with email systems.

x??

---

#### Capability Extension
Background context: Capability extension tools are designed to address inherent limitations of AI models, such as difficulty in performing certain tasks like arithmetic or programming. These tools can significantly enhance an agent's performance by providing necessary utilities directly.

:p What is a code interpreter and how can it be used?
??x
A code interpreter is a tool that allows agents to execute code snippets, return results, or analyze failures. This capability enables the agent to act as:
- A coding assistant.
- A data analyst.
- A research assistant capable of running experiments and reporting results.

Example usage in pseudocode:
```python
def execute_code(code):
    # Execute provided code snippet
    result = eval(code)
    return result

# Example call to function
result = execute_code("2 + 2")
print(result)  # Output: 4
```

x??

---

#### Write Actions and Tool Inventory
Background context: Write actions allow tools to modify data sources within the environment. This is in contrast to read-only actions, which only provide information.

:p How do write actions differ from read-only actions?
??x
Write actions involve making changes to the data sources, while read-only actions only allow reading from them. For example:
- A SQL executor can retrieve a table (read) but also change or delete the table (write).

Example of using a tool for write actions in pseudocode:
```sql
UPDATE customers SET email = 'new.email@example.com' WHERE id = 1;
DELETE FROM orders WHERE order_date < '2023-01-01';
```

x??

---

#### Email and Banking API Capabilities
Background context: APIs can perform various actions such as reading emails, responding to them, retrieving bank balances, and initiating transfers. These capabilities enable a system to automate customer outreach workflows and handle complex tasks efficiently.

:p How do email and banking APIs differ in their functions?
??x
Email APIs can read an email and respond to it, whereas banking APIs can retrieve your current balance and initiate a bank transfer. This distinction is crucial because while reading emails or retrieving balances are non-intrusive actions, initiating transfers involve financial transactions that require higher security measures.
x??

---

#### Autonomous AI Agents
Background context: Autonomous AI agents can perform complex tasks like researching potential customers, drafting emails, sending first emails, and following up with responses. However, there is a risk associated with giving AI the authority to perform potentially harmful actions.

:p What are some valid concerns related to autonomous AI systems?
??x
Valid concerns include the manipulation of financial markets, theft of copyrights, violation of privacy, reinforcement of biases, spread of misinformation and propaganda, among others. These risks highlight the importance of ensuring safety and security in AI applications.
x??

---

#### Self-Driving Cars and Physical vs. Non-Physical Harm
Background context: The example of self-driving cars is used to illustrate how AI systems can cause harm physically or non-physically. While a hacked car could potentially result in physical harm, other forms of AI manipulation can lead to significant social and economic issues.

:p How does the self-driving car example relate to autonomous AI systems?
??x
The self-driving car example is often used to highlight the potential for physical harm caused by AI systems. However, it's important to recognize that non-physical harms such as financial fraud, copyright theft, privacy violations, bias reinforcement, and misinformation can also be significant issues.
x??

---

#### Function Calling with Models
Background context: Model providers support function calling, which allows models to use external tools for various tasks. This feature is expected to become more common in the future.

:p What does function calling enable in model usage?
??x
Function calling enables models to interact with external tools and perform a wide range of tasks, from scheduling trips to processing financial transactions. This capability enhances the model's ability to solve complex real-world problems.
x??

---

#### Planning in Foundation Model Agents
Background context: Complex tasks require planning, which involves understanding the task, considering different options, and choosing the most promising approach.

:p What is the role of planning in foundation model agents?
??x
The role of planning is to outline a roadmap for accomplishing complex tasks. Effective planning requires models to understand the task, consider multiple options, and select the best course of action.
x??

---

#### Examples of Complex Tasks
Background context: A specific example given is scheduling a two-week trip from San Francisco to India with a budget constraint.

:p Can you provide an example of a complex task for a foundation model agent?
??x
An example of a complex task is scheduling a two-week trip from San Francisco to India with a budget of $5,000. The goal is the two-week trip, and the constraint is the budget.
x??

---

#### Planning as an Important Computational Problem
Background context: Planning is a well-studied computational problem that requires understanding and considering various steps to achieve a task.

:p Why is planning considered an important computational problem?
??x
Planning is considered an important computational problem because it involves breaking down complex tasks into manageable steps, evaluating different options, and selecting the most promising approach. This process is crucial for solving real-world problems effectively.
x??

---

#### Decoupling Planning and Execution

Planning involves generating a sequence of steps to achieve a goal, while execution entails carrying out those steps. Without proper validation, an agent might generate a long or even invalid plan that consumes resources without yielding results.

:p What is decoupling planning from execution in the context of agents?
??x
Decoupling planning and execution means first generating potential plans for how to achieve a goal, then validating these plans before executing them. This approach helps ensure that only reasonable and feasible plans are executed, saving time and resources.
For example, if an agent is tasked with finding companies without revenue but having raised at least $1 billion, it might generate a plan that first searches for all such companies (Option 1) or filters by raised capital then checks for non-revenue status (Option 2). Validating the plans using heuristics ensures more efficient execution.
```python
# Pseudocode to demonstrate plan generation and validation
def validate_plan(plan):
    # Example heuristic: eliminate plans requiring unavailable tools
    if "google_search" in plan.actions and not has_google_access():
        return False
    # Check length of the plan
    if len(plan.steps) > MAX_STEPS:
        return False
    return True

# Generate and execute a validated plan
plan = generate_plan()
if validate_plan(plan):
    execute_plan(plan)
else:
    print("Invalid or inefficient plan generated.")
```
x??

---

#### Validation Heuristics

Validation heuristics are used to ensure that plans generated by an agent are reasonable and feasible. Common heuristics include checking for the availability of required tools, ensuring plans do not exceed a certain step count, and evaluating the overall reasonableness of the plan.

:p What are validation heuristics in the context of planning?
??x
Validation heuristics are criteria or rules used to evaluate the validity and efficiency of generated plans. They help filter out invalid or inefficient plans before execution. Examples include checking for required tools (e.g., Google Search access) and ensuring the plan does not exceed a certain number of steps.

For instance, if a plan requires a Google search but your system lacks this capability, the heuristic will eliminate such a plan as it is impractical to execute.
```python
# Example validation heuristic function in pseudocode
def check_tools(plan):
    required_tools = {"google_search"}
    available_tools = get_available_tools()
    return all(tool in available_tools for tool in required_tools)

def validate_plan_length(plan, max_steps=1000):
    return len(plan.steps) <= max_steps

# Apply validation heuristics
if check_tools(plan) and validate_plan_length(plan):
    execute_plan(plan)
else:
    print("Plan not validated.")
```
x??

---

#### Multi-Agent System Components

A multi-agent system consists of several components: one for generating plans, another for validating them, and a third for executing those plans. Each component can be considered an agent in its own right, leading to complex interactions and workflows.

:p How does the structure of a multi-agent system work?
??x
In a multi-agent system, the overall task is broken down into smaller subtasks that are handled by specialized agents. There are three primary components:

1. **Plan Generation Agent**: Generates potential plans for achieving the goal.
2. **Validation Agent**: Evaluates generated plans using heuristics to ensure they are valid and efficient.
3. **Execution Agent**: Executes validated plans.

This structure allows each agent to focus on its specific task, potentially speeding up the process by generating multiple plans in parallel and evaluating them simultaneously.

For instance:
```python
# Pseudocode for multi-agent system
class PlanGenerator:
    def generate_plan(self):
        # Generate a plan based on some criteria
        pass

class Validator:
    def validate_plan(self, plan):
        # Validate the plan using heuristics
        return True  # Placeholder logic

class Executor:
    def execute_plan(self, plan):
        # Execute the validated plan
        pass

# Example workflow
generator = PlanGenerator()
validator = Validator()
executor = Executor()

plan = generator.generate_plan()
if validator.validate_plan(plan):
    executor.execute_plan(plan)
else:
    print("Plan not valid.")
```
x??

---

#### Parallel Plan Generation and Evaluation

To speed up the process, multiple plans can be generated in parallel. An evaluator then selects the most promising plan for execution. This approach introduces a trade-off between latency (time to complete) and cost (resources consumed during evaluation).

:p How does generating multiple plans in parallel improve an agent's workflow?
??x
Generating multiple plans in parallel allows for faster identification of promising plans, potentially reducing overall processing time. However, this comes at the cost of increased resource consumption since more plans are being generated simultaneously.

For example, if your system needs to find companies that have raised over $1 billion but do not generate revenue, it can generate several different strategies (plans) in parallel and evaluate them concurrently using a validator. The most promising plan is then selected for execution.
```python
# Pseudocode for parallel plan generation and evaluation
def generate_and_validate_plans(parallel_plan_count):
    plans = []
    results = []
    
    for _ in range(parallel_plan_count):
        generated_plan = PlanGenerator().generate_plan()
        validated_plan = Validator().validate_plan(generated_plan)
        
        if validated_plan:
            results.append((generated_plan, validate_plan(generated_plan)))
    
    # Select the most promising plan
    best_plan = max(results, key=lambda x: x[1])
    Executor().execute_plan(best_plan)

# Example usage
generate_and_validate_plans(5)  # Generate and evaluate 5 plans in parallel
```
x??

---

#### Intent Classification Mechanism
Intent classification helps agents understand the user's query to determine appropriate actions. This mechanism can be considered another agent within a multi-agent system, assisting with task decomposition and tool selection based on the intent.

:p What is the role of intent classification in an agent system?
??x
Intent classification plays a crucial role in guiding the correct course of action by categorizing user queries into predefined intents. For instance, if a query pertains to billing information, the intent classifier might classify it as 'billing inquiry', enabling the agent to retrieve relevant data or documentation.

For example:
```python
# Pseudocode for an Intent Classifier
def classify_intent(query):
    # Dummy classification logic
    if "payment" in query.lower():
        return "billing_inquiry"
    elif "password" in query.lower():
        return "password_reset"
    else:
        return "irrelevant"
```
x??

---

#### Planning Processes
Solving a task typically involves several stages: plan generation, reflection and error correction, execution, and further reflection and error correction upon receiving outcomes. These steps are essential for ensuring the goal is achieved accurately.

:p What are the main processes involved in solving a task?
??x
The main processes involve:
1. **Plan Generation**: Creating a sequence of actions to accomplish the task.
2. **Reflection and Error Correction**: Evaluating the generated plan and making adjustments if necessary.
3. **Execution**: Carrying out the planned actions.
4. **Post-Execution Reflection and Error Correction**: Assessing the outcomes and determining if the goal was met, then generating a new plan if needed.

For example:
```python
def solve_task(task):
    # Plan generation
    action_sequence = generate_plan(task)
    
    # Execution
    outcome = execute_plan(action_sequence)
    
    # Post-execution reflection
    if not is_goal_achieved(outcome):
        new_plan = correct_and_generate_new_plan(action_sequence, outcome)
        outcome = execute_plan(new_plan)
```
x??

---

#### Foundation Models as Planners
The ability of foundation models, particularly autoregressive language models, to plan remains an open question. Research suggests that these models are not capable of planning due to their structure and limitations.

:p How do researchers view the planning capabilities of foundation models?
??x
Researchers such as Yann LeCun from Meta have stated unequivocally that autoregressive LLMs cannot plan. S. Kambhampati argues that while LLMs are proficient at extracting knowledge, they struggle with creating executable plans. The core issue lies in the nature of planning, which involves searching through different paths to a goal and predicting outcomes.

For example:
```python
# Pseudocode illustrating why autoregressive models might fail in planning
def generate_plan(task):
    # Dummy function that fails to provide an actionable plan due to its structure
    return None

# This would indicate the model's inability to handle complex planning tasks.
```
x??

---

#### Decomposition of Tasks
Decomposing a task involves breaking it down into manageable actions, which is crucial for effective planning. This process helps in understanding the complexity and identifying the steps needed to achieve the goal.

:p How does decomposition help in solving complex tasks?
??x
Decomposition simplifies complex tasks by breaking them down into smaller, more manageable actions. By doing so, it makes the task easier to understand and execute, reducing the cognitive load on both humans and AI systems. For example, if a user wants to automate a report generation process, decomposition might involve:
1. Collecting data.
2. Cleaning the data.
3. Processing the data.
4. Generating the report.

This breakdown allows for clear steps that can be automated or executed manually as needed.

For example:
```python
def decompose_task(task):
    actions = []
    if "collect" in task.lower():
        actions.append("Collect relevant data")
    if "clean" in task.lower():
        actions.append("Clean the collected data")
    # Add more steps as necessary
    return actions

# Example usage
actions = decompose_task("Generate a report by cleaning and processing sales data.")
print(actions)
```
x??

---

#### Backtracking in Search
Backtracking is a fundamental technique used to find solutions by incrementally building candidates and abandoning a candidate ("backtracking") as soon as it is determined that the candidate cannot possibly be completed to a valid solution. In search problems, if an action does not lead to a promising state, backtracking allows revisiting previous decisions.
:p Explain the concept of backtracking in the context of searching?
??x
Backtracking involves exploring potential solutions incrementally and abandoning them if they are found to be non-promising. For instance, consider a scenario where you have two actions (A and B) at a given step. If taking action A leads to an unfruitful state, backtracking allows revisiting the previous state to take alternative action B.
x??

---

#### Autoregressive Models and Backtracking
Autoregressive models generate sequences of outputs in a sequential manner based on previously generated elements. Critics argue that these models cannot perform backtracking as they only generate forward actions, making them unsuitable for planning tasks. However, this limitation can be circumvented by revisiting previous states to consider alternative actions.
:p Can autoregressive models perform backtracking and planning?
??x
Autoregressive models can indeed perform backtracking and planning by revisiting previous states when an initially chosen path is deemed unproductive. By reassessing the situation, the model can choose a different course of action (e.g., from A to B), effectively performing backtracking.
x??

---

#### Planning with World Knowledge
Reasoning capabilities in models like LLMs allow them to predict outcomes of actions based on their vast world knowledge. This capability enables coherent planning by incorporating predicted outcomes into the sequence of actions, potentially leading to more effective plans.
:p How can LLMs generate coherent plans?
??x
LLMs can generate coherent plans by leveraging their extensive world knowledge to predict the outcome of each action. By understanding potential states resulting from different actions, they can make informed decisions, leading to more coherent and effective planning.
x??

---

#### Agent Concept in Reinforcement Learning (RL)
Agents in RL are defined as entities that take actions in dynamic environments to maximize cumulative rewards. The primary difference between RL agents and foundation model (FM) agents lies in their planners: RL agents use an RL algorithm for training, while FM agents rely on the model itself.
:p What is a core concept in reinforcement learning?
??x
A core concept in reinforcement learning (RL) is the agent—a software entity that interacts with its environment by taking actions to maximize cumulative rewards. Agents are trained using algorithms like Q-learning or policy gradients to make optimal decisions.
x??

---

#### FM Agent vs RL Agent
Both foundation model (FM) agents and RL agents share similar characteristics regarding their environments and possible actions. However, they differ in how their planners operate: RL agents use an RL algorithm for training, whereas FM agents rely on the model itself. There is potential to integrate RL algorithms into FM agents to enhance performance.
:p How do FM agents and RL agents differ?
??x
FM agents and RL agents differ primarily in their planning mechanisms. FM agents utilize models as planners, which can be prompted or fine-tuned for better planning capabilities, requiring less training time and resources. In contrast, RL agents use an RL algorithm to train a planner, often necessitating more extensive training and computational resources.
x??

---

#### Plan Generation with Prompt Engineering
Prompt engineering is a method of turning models into plan generators by providing clear instructions and context. For instance, creating an agent for product learning at Kitty Vogue involves giving the agent access to tools like price retrieval, top products, and product information.
:p How can prompt engineering be used to generate plans?
??x
Prompt engineering can be used to generate plans by carefully crafting instructions that guide the model towards producing coherent action sequences. For example, in the context of Kitty Vogue, a well-crafted prompt could instruct the agent on how to use available tools (price retrieval, top products, etc.) to assist customers effectively.
x??

---

#### Plan Proposal System
Background context explaining the system prompt used for proposing plans. The prompt provides a structured way to generate actions and their sequences for various tasks, using predefined functions.

:p What is the purpose of the SYSTEM PROMPT provided in the example?
??x
The purpose of the SYSTEM PROMPT is to guide an AI agent in generating a sequence of valid actions (plans) that solve given tasks. The prompt specifies five available actions: `get_today_date()`, `fetch_top_products(start_date, end_date, num_products)`, `fetch_product_info(product_name)`, `generate_query(task_history, tool_output)`, and `generate_response(query)`.

```python
# Example of a plan generated by the agent following the prompt
plan = [
    fetch_product_info,
    generate_query,
    generate_response
]
```
x??

---

#### Task and Plan Examples
This section provides examples of tasks and their corresponding plans, illustrating how the actions are structured to solve different types of user inputs.

:p Can you explain how a plan is generated for the task "What’s the price of the best-selling product last week"?
??x
To generate a plan for the task "What’s the price of the best-selling product last week," the agent would follow these steps:
1. Determine the current date using `get_time()`.
2. Fetch the top products sold in the last week with `fetch_top_products()` by inferring appropriate start and end dates.
3. Retrieve detailed information about the first (best-selling) product from that list with `fetch_product_info()`.
4. Generate a query to extract the price of the best-selling product using `generate_query()`, utilizing the task history and recent tool outputs.
5. Generate a response to the user's question using `generate_response()`.

```python
# Example plan for the given task
plan = [
    get_time,
    fetch_top_products,  # Assuming last week is defined as start_date="2030-09-07" and end_date="2030-09-13"
    fetch_product_info,
    generate_query,
    generate_response
]
```
x??

---

#### Plan Generation Challenges
This section highlights the complexities in generating plans due to uncertain parameters and potential hallucinations (mistaken predictions) from AI models.

:p How do uncertainties affect the planning process?
??x
Uncertainties in the planning process arise because it is often unclear what exact parameter values should be used for functions based on the available information. For example, if a user asks "What’s the average price of best-selling products?", key questions such as "How many best-selling products does the user want to look at?" or "Does the user want the best-selling products from last week, last month, or all time?" remain ambiguous.

To handle these uncertainties, AI models may need to make guesses, which can lead to hallucinations. Hallucinations include calling invalid functions or valid functions with incorrect parameters.

```python
# Example of handling parameter uncertainty
def fetch_top_products(start_date=None, end_date=None, num_products=1):
    # Logic to fetch top products based on inferred dates and number
    pass

start_date = get_time() - timedelta(days=7)  # Assuming last week is defined as the past 7 days
end_date = get_time()
plan = [
    get_time,
    fetch_top_products(start_date=start_date, end_date=end_date),
    fetch_product_info,
    generate_query,
    generate_response
]
```
x??

---

#### Improving Planning Capabilities
This section discusses methods to enhance an agent's ability to generate effective plans.

:p How can we improve the performance of AI models in generating plans?
??x
Improving the planning capabilities of AI models can be achieved through several strategies:
1. **Better System Prompt:** Providing more detailed examples and clear descriptions in the system prompt helps the model understand task requirements better.
2. **Improved Tool Descriptions:** Offering clearer, more comprehensive descriptions of available tools and their parameters aids in accurate function calls.
3. **Function Refactoring:** Simplifying complex functions into simpler ones can make it easier for models to generate correct plans.
4. **Using Stronger Models:** More powerful models are generally better at planning tasks due to their enhanced capabilities.
5. **Fine-Tuning:** Specializing the model for plan generation through fine-tuning can improve its performance.

```python
# Example of a simplified function
def fetch_top_products(start_date, end_date):
    # Logic to fetch top products based on date range
    pass

plan = [
    get_time,
    fetch_top_products(start_date="2030-09-07", end_date="2030-09-13"),
    fetch_product_info,
    generate_query,
    generate_response
]
```
x??

---

#### Function Calling and Tool Use
Background context explaining how models can be turned into agents by using tools, which are essentially functions. Different APIs handle function calling differently but generally involve declaring a tool inventory, specifying the tools to use per query, and allowing various settings like `required`, `none`, or `auto` for tool selection.
:p What is function calling in the context of model APIs?
??x
Function calling in model APIs refers to the process where an AI agent selects and invokes functions (tools) from a predefined list based on user queries. The goal is to enhance the agent's ability to perform tasks by leveraging external functionalities provided as callable functions.
```python
# Example of declaring tools in pseudocode
tools = [
    {"name": "lbs_to_kg", "description": "Convert pounds to kilograms", "params": [{"name": "lbs", "type": "float"}]}
]
```
x??

---

#### Tool Inventory and Declaration
Context explaining the creation of a tool inventory, which is necessary for defining all potential tools that an AI agent might utilize. Each tool is described by its function name, parameters, and documentation.
:p What does creating a tool inventory involve in model APIs?
??x
Creating a tool inventory involves declaring each available tool with details such as its name, the parameters it requires, and its purpose or functionality. This declaration helps the AI agent understand what functions are available for use during task execution.

```python
# Example of tool declaration in pseudocode
tool_inventory = [
    {"name": "convert_currency", "description": "Converts currency amounts from one type to another", "params": [{"name": "amount", "type": "float"}, {"name": "from_currency", "type": "str"}, {"name": "to_currency", "type": "str"}]},
    {"name": "search_web", "description": "Performs a web search for the given query", "params": [{"name": "query", "type": "str"}]}
]
```
x??

---

#### Specifying Tools per Query
Explanation of how different queries might require different tools, and APIs allow specifying which tools to use based on settings like `required`, `none`, or `auto`.
:p How can an agent specify the tools it uses for a query?
??x
An agent can specify the tools it uses for a query by defining a list of tools from its inventory that are relevant to the user's request. Settings such as `required`, `none`, and `auto` help in deciding whether to use any tool or let the model decide on its own.

```python
# Example of specifying tools in pseudocode
query = "How many kilograms are 40 pounds?"
selected_tools = ["lbs_to_kg"]
```
x??

---

#### ModelResponse Structure
Description of how a model's response includes information about function calls, useful for generating accurate responses after executing the functions.
:p What does a typical `ModelResponse` look like when involving function calls?
??x
A typical `ModelResponse` that involves function calls typically includes details about the function name and parameters. This structure helps in understanding how the model has processed the request and what actions it intends to perform.

```python
# Example of ModelResponse with tool calls in pseudocode
response = {
    "finish_reason": "tool_calls",
    "message": {
        "content": None,
        "role": "assistant",
        "tool_calls": [
            {"function": {"arguments": "{\"lbs\": 40}", "name": "lbs_to_kg"}, "type": "function"}
        ]
    }
}
```
x??

---

#### Planning Granularity
Explanation of planning granularity and the trade-off between detailed and high-level plans, as well as how hierarchical planning can address this issue.
:p What is planning granularity in the context of task execution?
??x
Planning granularity refers to the level of detail at which a plan is created. A higher-granularity plan provides more specific steps for executing a task, while a lower-granularity plan offers broader outlines. There's often a trade-off between the effort required to generate detailed plans versus the ease with which they can be executed.

Hierarchical planning involves creating high-level plans first and then refining them into more detailed sub-plans as needed. This approach balances the complexity of generating precise plans with the practicality of executing them.
```python
# Example of hierarchical planning in pseudocode
high_level_plan = {"quarters": ["plan_q1", "plan_q2"]}
sub_plans = {
    "plan_q1": {"months": ["plan_jan", "plan_feb", "plan_mar"]},
    "plan_q2": {"months": ["plan_apr", "plan_may", "plan_jun"]}
}
```
x??

---

#### Changing Tool Inventory Over Time
Background context explaining how tool names can change over time, such as renaming `get_time()` to `get_current_time()`. This necessitates updating prompts and examples, as well as retraining models if they were finetuned on a specific set of tools. Using more natural language plans is proposed as a solution to mitigate these issues.
:p How does changing tool names affect the planning process?
??x
When tool names change over time (e.g., `get_time()` renamed to `get_current_time()`), it requires updating all existing prompts and examples in the planner system, and potentially retraining models that were fine-tuned on the old inventory. Using natural language plans can help make the system more robust against such changes.
x??

---

#### Natural Language Plans vs. Domain-Specific Function Names
Background context explaining why using domain-specific function names can lead to issues when tools change. It discusses how natural language plans are less prone to hallucination but require a translator for execution.
:p Why might it be better to use natural language for plans?
??x
Using natural language for plans is beneficial because it makes the planner more robust to changes in tool APIs and reduces the risk of hallucinations, as the model can focus on understanding high-level instructions rather than specific function names. However, this approach necessitates a translator that converts natural language actions into executable commands.
x??

---

#### Translator Role
Background context explaining the role of a translator in converting natural language plans to executable commands. It notes that translating is simpler and less prone to hallucinations compared to planning.
:p What is the translator's function?
??x
The translator’s function is to convert high-level, natural language actions into executable commands. This process is generally simpler and carries a lower risk of hallucination compared to generating complex plans directly from natural language.
x??

---

#### Complex Plans Overview
Background context explaining different types of control flows (sequential, parallel, if statement, for loop) in the context of planning. These control flows determine the order in which actions can be executed.
:p What are the different types of control flows mentioned?
??x
The text mentions four types of control flows: 
- Sequential: Task B is executed after task A is complete.
- Parallel: Tasks A and B are executed simultaneously.
- If statement: A decision-based flow where a condition determines whether to execute task B or C.
- For loop: Repeated execution of a task until a specific condition is met.

These control flows help in determining the order and logic of actions within plans.
x??

---

#### Sequential Control Flow Example
Background context explaining sequential control flow, where tasks are executed one after another because they depend on each other. It provides an example with SQL queries.
:p What is an example of a task following a sequential control flow?
??x
An example of sequential control flow is when executing an SQL query depends on the completion of a previous step. For instance, a natural language input might require translating into SQL first before running it.

```java
// Pseudocode for sequential execution
String sqlQuery = translateNaturalLanguageToSQL(naturalInput);
executeSQL(sqlQuery);
```
x??

---

#### Parallel Control Flow Example
Background context explaining parallel control flow, where multiple tasks can be executed simultaneously. It provides an example with best-selling products under $100.
:p What is an example of a task following a parallel control flow?
??x
An example of parallel control flow is when retrieving the top 100 best-selling products and checking their prices are done concurrently, as shown below:

```java
// Pseudocode for parallel execution
List<Product> bestSellingProducts = retrieveTop100BestSellingProducts();
for (Product product : bestSellingProducts) {
    retrievePrice(product);
}
```
x??

---

#### If Statement Control Flow Example
Background context explaining if statement control flow, where the next action depends on a condition from the previous step. It provides an example with stock decisions.
:p What is an example of a task following an if statement control flow?
??x
An example of if statement control flow is when checking a condition (e.g., NVIDIA's earnings report) to decide between two actions:

```java
// Pseudocode for if statement execution
EarningsReport report = checkNVIDIAEarnings();
if (report.isPositive()) {
    buyNVIDIAStocks();
} else {
    sellNVIDIAStocks();
}
```
x??

---

#### For Loop Control Flow Example
Background context explaining for loop control flow, where a task is repeated until a specific condition is met. It provides an example with generating random numbers.
:p What is an example of a task following a for loop control flow?
??x
An example of for loop control flow is when generating random numbers until finding a prime number:

```java
// Pseudocode for for loop execution
int randomNumber;
do {
    randomNumber = generateRandomNumber();
} while (!isPrime(randomNumber));
```
x??

---

#### Parallel Execution Support
Background context: The ability to execute tasks simultaneously can significantly reduce latency and enhance user experience, especially when dealing with tasks that do not rely on sequential execution. For example, browsing ten websites at once is more efficient than sequentially opening them one by one.

:p Can an agent framework support parallel execution?
??x
Yes, an agent framework should be capable of supporting parallel execution to handle tasks that can run concurrently without dependencies. This feature helps in reducing the overall time taken for executing such tasks, thereby improving user experience.
For instance, if a task involves fetching data from multiple sources, these fetches could be performed simultaneously rather than sequentially.

```java
// Pseudocode example of parallel execution using Java's CompletableFuture
import java.util.concurrent.CompletableFuture;

public class ParallelTaskExecutor {
    public void executeTasks() {
        // Assume tasks are represented as functions that return a value when completed
        CompletableFuture<String> task1 = CompletableFuture.supplyAsync(() -> fetchDataFromSource1());
        CompletableFuture<String> task2 = CompletableFuture.supplyAsync(() -> fetchDataFromSource2());

        // When both tasks are complete, perform some action
        task1.thenCombine(task2, (result1, result2) -> processResults(result1, result2));
    }

    private String fetchDataFromSource1() {
        // Simulate fetching data from a source
        return "Data from Source 1";
    }

    private String fetchDataFromSource2() {
        // Simulate fetching data from another source
        return "Data from Source 2";
    }

    private void processResults(String result1, String result2) {
        // Process the results of both tasks
        System.out.println("Processing results: " + result1 + ", " + result2);
    }
}
```
x??

---
#### Reflection Mechanism in Agents
Background context: Reflection is a critical component for agents to evaluate and adjust their plans during task execution. It helps in identifying errors, understanding the feasibility of user queries, and ensuring successful task completion.

:p What is reflection, and why is it important in agent frameworks?
??x
Reflection in agent frameworks involves mechanisms where an agent evaluates its actions and decisions throughout a task process. This self-assessment can help identify errors, validate plans, and ensure that the agent stays on track to achieve its goals.
Reflection is crucial because even well-planned tasks may encounter unexpected issues or require adjustments based on the environment or user feedback.

```java
// Example of an agent generating thought-act-reflection steps
public class ReflectiveAgent {
    public void executeTask() {
        // Initial plan and action
        String act1 = "Visit website 1";
        System.out.println("Thought: Planning to visit website 1");
        System.out.println("Act: Visiting website 1");

        // Simulate an observation or feedback
        boolean success = true; // Assume the operation was successful

        // Evaluate the action based on the outcome
        if (success) {
            System.out.println("Observation: Successfully visited website 1");
            System.out.println("Thought: The plan seems feasible, continue to next step.");
        } else {
            System.out.println("Observation: Could not visit website 1");
            // Reflect and adjust the plan
            String newAct = "Try visiting a different site";
            System.out.println("Thought: Attempting a different action due to failure.");
            System.out.println("Act: Trying to visit a different site");
        }
    }
}
```
x??

---
#### Interleaving Reasoning and Action (ReAct Framework)
Background context: The ReAct framework proposes interleaving reasoning with action in an agent's workflow. This approach, where the agent reflects on its actions after each step or after a series of steps, allows for dynamic adjustments based on outcomes.

:p How does the ReAct framework enhance an agent’s task execution?
??x
The ReAct framework enhances an agent's task execution by incorporating continuous reasoning and reflection into the workflow. At each step, the agent evaluates its actions, adjusts its plans if necessary, and makes informed decisions to ensure successful task completion.
This method helps in identifying errors early, making corrections, and adapting strategies based on real-time feedback.

```java
// Example of a ReAct-like framework for an agent
public class ReactiveAgent {
    public void executeTask() {
        // Initial plan
        String initialPlan = "Gather information from source A";

        while (!isTaskComplete()) {
            System.out.println("Thought: Planning to gather info from source A");
            System.out.println("Act: Gathering info from source A");

            // Simulate action and observe the result
            boolean success = false; // Assume an initial failure

            if (success) {
                System.out.println("Observation: Successfully gathered information.");
                System.out.println("Thought: Task is on track, continue to next step.");
            } else {
                System.out.println("Observation: Failed to gather info from source A");
                // Reflect and adjust the plan
                String newPlan = "Gather info from an alternative source B";
                System.out.println("Thought: Adjusting strategy due to initial failure. New plan is gathering info from source B.");
                System.out.println("Act: Gathering info from source B");
            }
        }

        // Final reflection and completion check
        System.out.println("Reflection: Task completed successfully based on final outcome checks.");
    }

    private boolean isTaskComplete() {
        // Placeholder method to simulate task completion condition
        return true;
    }
}
```
x??

---

#### Reflexion Framework Overview
Reflexion, as described by Shinn et al., 2023, is a method that splits reflection into two modules: an evaluator and a self-reflection module. The evaluator assesses outcomes while the self-reflection module analyzes what went wrong. This framework is applied in agents to continuously refine their actions based on feedback.
:p What does Reflexion do?
??x
Reflexion separates reflection into an evaluator that evaluates outcomes and a self-reflection module that analyzes why certain actions were not successful. This allows for continuous improvement by refining trajectories after each step.
x??

---

#### Trajectory in Reflexion
In Reflexion, the term "trajectory" is used to refer to a plan or strategy that the agent follows. After evaluation and self-reflection, the agent proposes a new trajectory at each step.
:p What does the term "trajectory" mean in Reflexion?
??x
The term "trajectory" refers to a plan or strategy that an agent follows during its task execution. After evaluating outcomes and reflecting on what went wrong, the agent suggests a new plan for the next step.
x??

---

#### Performance Improvement from Reflection
Reflection can significantly improve performance with relatively low implementation difficulty compared to generating plans. However, it comes with the downsides of increased latency and cost due to token-intensive processes.
:p What are the benefits and drawbacks of using reflection in agents?
??x
Benefits include surprisingly good performance improvement at a lower implementation effort compared to plan generation. Drawbacks involve higher latency and costs because generating thoughts, observations, and sometimes actions require many tokens, which can increase both time and financial costs, especially for tasks with multiple steps.
x??

---

#### Tool Selection for Agents
The selection of tools is critical in task success but depends on the environment, task, and AI model used. The more tools an agent has, the more complex it becomes to use them effectively. Experimentation and analysis are necessary to determine which set of tools works best.
:p What factors should be considered when selecting tools for agents?
??x
Factors include the nature of the environment, the specific task requirements, and the AI model capabilities. More tools can enhance an agent's capability but increase complexity in their efficient use. Experimentation with different sets of tools, ablation studies to identify essential tools, and analyzing tool call distributions are useful methods.
x??

---

#### Example: Tool Use Patterns
The differences in tool usage between GPT-4 and ChatGPT were studied by Chameleon (Lu et al., 2023), showing varied patterns based on model type and task complexity. This study highlights the importance of understanding how tools are used across different models.
:p What did the Chameleon study reveal about tool use?
??x
The Chameleon study revealed that GPT-4 and ChatGPT exhibit different patterns in tool usage, emphasizing the impact of model type and task characteristics on tool selection and application. This highlights the need for tailored approaches to tool integration depending on the specific AI models used.
x??

---

#### Different Tools for Different Tasks

Background context: The text highlights that different tasks require different tools and that different models have varying preferences for these tools. For example, science question answering (ScienceQA) relies more on knowledge retrieval tools compared to tabular math problem-solving (TabMWP).

:p How do different tasks influence the choice of tools?
??x
Different tasks may require specific types of tools due to their nature and complexity. ScienceQA benefits from tools that can efficiently retrieve and process scientific information, whereas TabMWP might need tools for numerical computation and data manipulation.

For example:
```java
// Pseudocode for a tool selection strategy based on task type
public class ToolSelector {
    public Tool selectTool(Task task) {
        if (task instanceof ScienceQA) {
            return new KnowledgeRetrievalTool();
        } else if (task instanceof TabMWP) {
            return new DataManipulationTool();
        }
        // Other task types and corresponding tools can be added
        return null;
    }
}
```
x??

---

#### Different Models Have Different Tool Preferences

Background context: The text mentions that different models like GPT-4 and ChatGPT have different preferences for the tools they use. For instance, GPT-4 might prefer knowledge retrieval tools more than ChatGPT.

:p How do different AI models exhibit different tool preferences?
??x
Different AI models may be designed or trained to prioritize certain types of tools based on their architecture, training data, and intended use cases. This preference can affect how they interact with the environment and solve tasks.

For example:
```java
// Pseudocode showing a model's tool selection process
public class Model {
    private Map<String, Tool> preferredTools;

    public Tool selectTool(String task) {
        if (task.equals("knowledge retrieval")) {
            return preferredTools.get("knowledge retrieval");
        } else if (task.equals("image captioning")) {
            return preferredTools.get("image processing");
        }
        // Other tasks and corresponding tools
        return null;
    }

    public void setPreferredTool(String taskType, Tool tool) {
        this.preferredTools.put(taskType, tool);
    }
}
```
x??

---

#### Evaluating Agent Frameworks

Background context: The text discusses the importance of evaluating agent frameworks based on what planners and tools they support. Different frameworks might focus on different categories of tools.

:p How should one evaluate an agent framework?
??x
Evaluating an agent framework involves assessing its tool inventory, planner capabilities, and ease of extending to new tools. You need to consider whether the framework supports a wide range of tools relevant to your needs and how easily you can integrate new tools into the system.

For example:
```java
// Pseudocode for evaluating an agent framework
public class AgentFrameworkEvaluator {
    private Map<String, Tool> toolInventory;
    private Planner planner;

    public void evaluateToolSupport() {
        // Check if all required tools are present in the inventory
        boolean supportsAllTools = true;
        for (String toolName : requiredTools) {
            if (!toolInventory.containsKey(toolName)) {
                supportsAllTools = false;
                break;
            }
        }

        System.out.println("Tool Support: " + (supportsAllTools ? "Good" : "Poor"));
    }

    public void evaluatePlanner() {
        // Test planner's effectiveness in solving a task
        boolean planIsEffective = planner.plan(task).isSolvable();
        System.out.println("Planner Effectiveness: " + (planIsEffective ? "Good" : "Poor"));
    }
}
```
x??

---

#### Tool Transition

Background context: The text introduces the concept of tool transition, where an agent can learn to combine tools into more complex ones. This is useful for building progressively more powerful tools from simpler ones.

:p What is tool transition in the context of AI agents?
??x
Tool transition refers to the process by which an AI agent learns to combine and use different tools together to form more complex functionalities. This allows the agent to create new, advanced capabilities from basic tools it initially has access to.

For example:
```java
// Pseudocode for a tool transition mechanism
public class ToolTransition {
    private Map<Tool, List<Tool>> toolDependencies;

    public void learnTransition(Tool toolX, Tool toolY) {
        // Simulate learning the dependency between tool X and Y
        if (toolX.dependsOn(toolY)) {
            toolDependencies.put(toolX, Collections.singletonList(toolY));
        }
    }

    public List<Tool> getDependencyChain(Tool startTool) {
        return toolDependencies.get(startTool);
    }
}
```
x??

---

#### Skill Manager for Agents

Background context: The text describes a skill manager that tracks new skills (tools) acquired by an agent and stores them in a library. These skills can be reused later.

:p How does the skill manager work?
??x
The skill manager keeps track of new skills or tools that agents acquire during task execution. It assesses whether these newly created skills are useful and, if so, adds them to a library for future reuse. This helps in maintaining an agent's knowledge base without manual intervention.

For example:
```java
// Pseudocode for the skill manager
public class SkillManager {
    private Map<String, Tool> toolLibrary;

    public void addTool(Tool newTool) {
        if (isSkillful(newTool)) {
            toolLibrary.put(newTool.getName(), newTool);
        }
    }

    public boolean isSkillful(Tool tool) {
        // Logic to determine if the tool has been useful
        return true; // Placeholder logic
    }
}
```
x??

