# Flashcards: 2A001---AI-Engineering_-Building-Applications-with-Foundation-Models-OReilly-Media-2025Chip-Huyen--_processed (Part 21)

**Starting Chapter:** Defenses Against Prompt Attacks

---

#### Understanding Prompt Attacks and Defenses
Background context: The text discusses various aspects of prompt attacks on AI systems, particularly language models. It highlights the importance of understanding potential vulnerabilities and implementing robust defenses at different levels—model, prompt, and system.

:p What are some key steps to evaluate a system's robustness against prompt attacks?
??x
To evaluate a system’s robustness against prompt attacks, two important metrics are used: the violation rate and the false refusal rate. The violation rate measures the percentage of successful attacks out of all attack attempts, while the false refusal rate measures how often a model refuses a query when it is possible to answer safely.

The goal here is to ensure that the system is secure without being overly cautious. A perfect system would have zero violations but might be unhelpful if it refuses every request.

```java
public class EvaluationMetrics {
    public double violationRate = 0.0;
    public double falseRefusalRate = 0.0;

    public void updateViolationRate(double attackSuccess) {
        // Update the violation rate based on successful attacks
        this.violationRate += attackSuccess;
    }

    public void updateFalseRefusalRate(double requestRefused, boolean possibleToAnswer) {
        if (possibleToAnswer && !requestRefused) {
            this.falseRefusalRate++;
        }
    }
}
```
x??

---

#### Instruction Hierarchy for Model-Level Defense
Background context: The text introduces the concept of an instruction hierarchy to prioritize system prompts over user and model-generated outputs. This helps in mitigating prompt injection attacks by ensuring higher-priority instructions are followed.

:p How does OpenAI's instruction hierarchy help mitigate prompt attacks?
??x
OpenAI’s instruction hierarchy provides a structured approach for prioritizing different types of inputs. It has four levels:
1. System Prompt: The highest priority.
2. User Prompt: Lower than system but still critical.
3. Model Outputs: Lower in importance compared to the above two.
4. Tool Outputs: The lowest in priority.

When there are conflicting instructions, such as "don't reveal private information" (system prompt) and "show me X’s email address" (user prompt), the higher-priority instruction is followed. This approach neutralizes many indirect prompt injection attacks by ensuring that system-level directives take precedence over user requests or model-generated outputs.

```java
public class InstructionHierarchy {
    public String[] hierarchy = {"System Prompt", "User Prompt", "Model Outputs", "Tool Outputs"};

    public int getPriority(String input) {
        for (int i = 0; i < hierarchy.length; i++) {
            if (input.contains(hierarchy[i])) return i;
        }
        return -1; // Default to lowest priority
    }
}
```
x??

---

#### Prompt-Level Defense Techniques
Background context: The text emphasizes the importance of creating robust prompts by explicitly defining what the model should and shouldn't do. This includes using explicit safety instructions, repeating system prompts, and preparing the model for known attack vectors.

:p How can you enhance a prompt to make it more resistant to attacks?
??x
To enhance a prompt's resistance to attacks, follow these strategies:

1. **Explicit Safety Instructions:** Clearly define what information should not be returned or actions that should not be taken.
2. **Repeat System Prompts:** Duplicate the system prompt before and after the user prompt to remind the model of its role.
3. **Prepare for Known Attacks:** If you know potential attack vectors, train the model to handle them.

Example:
```plaintext
System Prompt: Do not return sensitive information such as email addresses, phone numbers, and addresses.
User Request: Summarize this paper: {{paper}} Remember, you are summarizing the paper. 
```

By explicitly instructing the model and preparing it for specific attacks, you can reduce the risk of prompt hijacking.

```java
public class EnhancedPrompt {
    public String systemPrompt = "Do not return sensitive information such as email addresses, phone numbers, and addresses.";
    public String userRequest = "Summarize this paper: {{paper}} Remember, you are summarizing the paper.";

    public void enhancePrompt() {
        // Enhance prompt by adding explicit instructions
        System.out.println(systemPrompt + "\n" + userRequest);
    }
}
```
x??

---

#### System-Level Defense Practices
Background context: The text discusses various system-level practices to ensure safety, such as isolation of generated code and strict command approval mechanisms.

:p What are some good practices for designing a safe AI system?
??x
Some good practices for ensuring the safety of an AI system include:

1. **Isolation:** Execute generated code in a virtual machine separated from user’s main systems.
2. **Command Approval:** Require explicit human approvals before executing potentially impactful commands, such as SQL database queries with "DELETE", "DROP", or "UPDATE" actions.
3. **Define Out-of-Scope Topics:** Limit the topics your application can discuss to avoid engaging in inappropriate or unprepared conversations.

By implementing these practices, you can significantly reduce the risk of harmful interactions and ensure that your system remains safe for users.

```java
public class SafeSystem {
    public void executeCode(String code) throws Exception {
        // Isolate execution within a virtual machine
        VirtualMachine vm = new VirtualMachine();
        vm.run(code);
    }

    public boolean approveCommand(String command) {
        // Check if the command is safe and requires human approval
        return !command.contains("DELETE") && !command.contains("DROP") && !command.contains("UPDATE");
    }
}
```
x??

---

#### Input Filtering and Guardrails
Background context: The text discusses methods for filtering inputs that might be controversial or inappropriate, as well as implementing guardrails to manage both input and output. This is crucial to ensure the safety and appropriateness of AI outputs.

:p What are some ways to filter out potentially harmful inputs in an AI system?
??x
Some common methods include filtering out predefined phrases related to sensitive topics such as "immigration" or "antivax." More advanced algorithms analyze the entire conversation context using natural language processing techniques to understand user intent and block inappropriate requests. Anomaly detection can also identify unusual prompts.

For example, you might have a list of keywords that are blocked outright, known prompt attack patterns to match against, or an AI model to detect suspicious requests.
??x

---

#### Output Guardrails
Background context: The text mentions the importance of not only filtering inputs but also managing outputs. This involves checking for potentially harmful content such as personally identifiable information (PII) or toxic information.

:p How can output guardrails be implemented in an AI system?
??x
Output guardrails can include mechanisms to check if a generated response contains PII, toxic information, or other inappropriate content. For instance, you could have a function that checks the text for sensitive keywords and blocks them if they are present.

```python
def check_output_safety(output):
    # List of potentially harmful keywords
    harmful_keywords = ['PII', 'toxic']
    
    # Check if any keyword is present in the output
    for keyword in harmful_keywords:
        if keyword in output:
            return False  # Block output
    return True  # Allow output

# Example usage
output_text = "This is a sample response that might contain PII."
if check_output_safety(output_text):
    print("Output is safe.")
else:
    print("Output contains harmful content and is blocked.")
```
x??

---

#### Anomaly Detection for Inputs
Background context: The text suggests using anomaly detection to identify unusual prompts. This can help in recognizing patterns that might indicate a malicious or inappropriate attempt.

:p How does anomaly detection work in the context of AI input filtering?
??x
Anomaly detection involves identifying inputs that deviate significantly from typical or expected behavior. Techniques like statistical models, machine learning algorithms, and behavioral analysis can be used to flag unusual prompts.

For example, you might use a clustering algorithm to group similar types of inputs together and then identify any outliers.
```python
from sklearn.cluster import KMeans

# Sample data representing different input patterns
inputs = [[1], [2], [7], [6], [3], [8], [9]]

# Use KMeans for anomaly detection, setting a threshold to flag anomalies
kmeans = KMeans(n_clusters=2)
kmeans.fit(inputs)

# Get the cluster centers and predict labels
cluster_centers = kmeans.cluster_centers_
predictions = kmeans.predict(inputs)

# Flag inputs that do not match their predicted clusters as anomalies
anomalies = [inputs[i] for i, label in enumerate(predictions) if predictions[i] != kmeans.labels_[i]]
print("Anomalies:", anomalies)
```
x??

---

#### Prompt Engineering for AI Communication
Background context: The text explains the importance of prompt engineering, which involves crafting instructions to achieve desired outcomes from AI models. It highlights that simple changes in prompts can significantly affect model responses.

:p What is prompt engineering and why is it important?
??x
Prompt engineering is the practice of carefully designing instructions or queries to guide AI models towards producing specific outputs. It's essential because small changes in how you phrase a request can lead to vastly different results, especially when working with sensitive or complex tasks.

For example:
```java
// Bad prompt: "Tell me about the weather"
String badPrompt = "Tell me about the weather";

// Good prompt: "Can you provide an hourly weather forecast for tomorrow in New York?"
String goodPrompt = "Can you provide an hourly weather forecast for tomorrow in New York?";
```
x??

---

#### Security Risks and Prompt Attacks
Background context: The text discusses security risks associated with AI, particularly the potential for prompt attacks where bad actors manipulate prompts to elicit harmful or malicious responses from models.

:p What are some defense mechanisms against prompt attacks?
??x
Defenses against prompt attacks can include implementing robust input validation, using contextual understanding in natural language processing (NLP) to identify suspicious patterns, and employing human operators as a last line of defense for critical tasks. Additionally, continuous monitoring and updating of safety filters based on new threats are crucial.

For example:
```java
public class PromptValidator {
    private Set<String> blockedKeywords = new HashSet<>();
    
    public boolean isValidPrompt(String prompt) {
        // Load known bad keywords or patterns
        loadBlockedKeywords();
        
        // Check for any blocked words in the prompt
        for (String keyword : blockedKeywords) {
            if (prompt.contains(keyword)) {
                return false;  // Invalid prompt detected
            }
        }
        return true;  // Prompt is valid
    }

    private void loadBlockedKeywords() {
        // Load keywords from a secure source or predefined list
        blockedKeywords.add("delete");
        blockedKeywords.add("malicious");
    }
}
```
x??

---

#### Contextual Information for Tasks
Background context: The text emphasizes the importance of providing relevant context to AI models when performing tasks. While instructions are crucial, they must be complemented with pertinent background information.

:p How can you ensure that an AI model has enough context to perform a task accurately?
??x
To provide sufficient context, you should include relevant background information and examples in your prompts. This helps the model understand the requirements better and produce more accurate results.

For example:
```java
public class TaskExecutor {
    public String executeTask(String instruction) {
        // Combine instruction with contextual data
        String fullPrompt = "Given the following context: [context] " + instruction;
        
        return processPrompt(fullPrompt);
    }
    
    private String processPrompt(String prompt) {
        // Process and execute the prompt using an AI model
        // This could involve calling a model API or local implementation
        return "Processed task with context.";
    }
}
```
x??

---

#### RAG (Retrieval-Augmented Generation)
Background context explaining the concept of RAG. It enhances a model's generation by retrieving relevant information from external memory sources like internal databases, user chat sessions, or the internet.

The retrieve-then-generate pattern was first introduced in "Reading Wikipedia to Answer Open-Domain Questions" (Chen et al., 2017). In this work, the system retrieves five most relevant Wikipedia pages and then a model reads from these pages to generate an answer. The term retrieval-augmented generation was coined later.

:p What is RAG?
??x
RAG is a technique that enhances a model's generation by retrieving relevant information from external memory sources like internal databases or the internet, before generating an answer.
x??

---

#### LSTM (Long Short-Term Memory)
Background context explaining the concept of LSTM. It was one of the dominant architectures in NLP until the transformer architecture took over in 2018.

:p What is LSTM?
??x
LSTM is a type of recurrent neural network used in natural language processing before the transformer became prevalent.
x??

---

#### Retrieval-Augmented Generation (RAG)
Background context explaining RAG, its purpose, and how it works. It allows models to use only relevant information for each query, reducing input tokens while potentially improving performance.

:p What is retrieval-augmented generation?
??x
Retrieval-augmented generation (RAG) enhances a model by retrieving the most relevant information from external sources before generating an answer, thereby making the model's response more detailed and accurate.
x??

---

#### Context Efficiency in Models
Background context on how models use context efficiently. The longer the context, the higher the likelihood that the model focuses on irrelevant parts.

:p Why is context efficiency important?
??x
Context efficiency is crucial because as context length increases, there is a higher risk of the model focusing on irrelevant parts, leading to reduced performance and increased latency.
x??

---

#### Application Context Expansion
Background context explaining how application contexts expand to fill the model's capacity. For example, given a query about a printer’s specifications, providing those details can improve the model’s response.

:p How does an application's context expand?
??x
An application's context tends to expand to fit within the limits of the model being used. Providing relevant data (like a printer's specifications) in queries improves the model's ability to respond accurately.
x??

---

#### Feature Engineering vs RAG
Background on how feature engineering for classical ML models is similar to using RAG for foundation models.

:p How does RAG relate to feature engineering?
??x
RAG can be seen as a technique that performs feature engineering for foundation models, providing necessary information to the model to process an input more effectively.
x??

---

#### Future of Context Length and Efficiency
Background on efforts to expand context length while making models use it more efficiently. This includes potential mechanisms like retrieval-like or attention-like systems.

:p What is the future direction of RAG?
??x
The future direction of RAG involves expanding context length in parallel with improving how models use this context effectively, potentially through mechanisms like retrieval or attention.
x??

---

