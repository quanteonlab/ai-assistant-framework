# Flashcards: 2B001--Financial-Data-Engineering_processed (Part 36)

**Starting Chapter:** What Is a Data Workflow

---

#### Workflow-Oriented Software Architectures (WOSA)
Background context: As companies grow, their technological stacks increase in size and complexity. This creates numerous interdependencies among system components, requiring organized software transactions into structured workflows that can be defined, coordinated, managed, monitored, and scaled following a specific business logic.

:p What is the purpose of Workflow-Oriented Software Architectures (WOSA)?
??x
The primary purpose of WOSAs is to manage complex processes by organizing them into structured workflows. This approach enhances modularity, efficiency, and manageability in software systems, ensuring that various system components can interact logically and follow specific business rules.

Code examples are less relevant here as it's more about the concept rather than implementation:
```java
// Example of defining a simple workflow in pseudocode
class Workflow {
    void defineSteps() {
        step1 = new Step("InitiateTrade");
        step2 = new Step("ExecuteTrade");
        step3 = new Step("CaptureTrade");

        step1.nextStep(step2);
        step2.nextStep(step3);
    }
}
```
x??

---

#### Data Workflow
Background context: A data workflow is a repeatable process that involves applying a structured sequence of data processing steps to an initial dataset, producing a desired output. These workflows are essential in financial data engineering and can be abstracted using the dataflow paradigm.

:p What defines a data workflow?
??x
A data workflow is defined as a repeatable process where a structured sequence of data processing steps is applied to an initial dataset, resulting in a desired output. This process often involves organizing tasks into a directed computational graph (DAG) with nodes representing individual computation steps and links expressing data dependencies.

Code examples are relevant here to illustrate the concept:
```java
// Example of defining a linear DAG in pseudocode
class DataWorkflow {
    Node task1 = new TaskNode("DataIngestion");
    Node task2 = new TaskNode("DataTransformation");
    Node task3 = new TaskNode("DataValidation");

    void createDAG() {
        task1.next(task2);
        task2.next(task3);
    }
}
```
x??

---

#### Linear Directed Acyclic Graph (DAG)
Background context: A linear DAG organizes tasks in a sequential order where each task N can have at most one preceding task (N–1) and one subsequent task (N+1). This structure is often used to represent simple data workflows.

:p What characterizes a linear Directed Acyclic Graph (DAG)?
??x
A linear Directed Acyclic Graph (DAG) organizes tasks in a sequential order where each task N can have at most one preceding task (N–1) and one subsequent task (N+1). In this structure, the graph does not contain any cycles.

Code examples are relevant to illustrate the concept:
```java
// Example of creating a linear DAG in pseudocode
class TaskNode {
    String name;
    TaskNode next;

    TaskNode(String name) {
        this.name = name;
    }

    void next(TaskNode task) {
        next = task;
    }
}

void createLinearDAG() {
    Node task1 = new TaskNode("DataIngestion");
    Node task2 = new TaskNode("DataTransformation");
    Node task3 = new TaskNode("DataValidation");

    task1.next(task2);
    task2.next(task3);
}
```
x??

---

#### Complex Directed Acyclic Graph (DAG)
Background context: In complex DAGs, a specific computing task may depend on multiple preceding tasks. This structure is more intricate than the linear case and requires careful design to ensure that data dependencies are properly managed.

:p What characterizes a complex Directed Acyclic Graph (DAG)?
??x
A complex Directed Acyclic Graph (DAG) allows for dependencies where a specific computing task can rely on multiple preceding tasks. Unlike a linear DAG, this structure is more intricate and requires careful design to ensure that data dependencies are properly managed without creating cycles.

Code examples are relevant:
```java
// Example of creating a complex DAG in pseudocode
class TaskNode {
    String name;
    List<TaskNode> predecessors;

    TaskNode(String name) {
        this.name = name;
        this.predecessors = new ArrayList<>();
    }

    void addPredecessor(TaskNode task) {
        predecessors.add(task);
    }
}

void createComplexDAG() {
    Node task1 = new TaskNode("DataIngestion");
    Node task2 = new TaskNode("DataTransformation");
    Node task3 = new TaskNode("DataValidation");

    // Complex dependency: Task 6 depends on tasks 3, 4, and 5
    Node task6 = new TaskNode("ComplexTaskDependentOnMultipleTasks");

    task1.addPredecessor(task2);
    task2.addPredecessor(task3);
    task3.addPredecessor(task4);
    task4.addPredecessor(task5);
    task5.addPredecessor(task6);
}
```
x??

---

#### Task Abstraction and Best Practices
Background context: When defining computational DAGs, it is essential to consider the nature of tasks. Tasks should be atomic units of execution that either succeed or fail as a whole. They should also be idempotent to allow for retry operations without causing side effects.

:p What are best practices when defining tasks in a data workflow?
??x
Best practices for defining tasks in a data workflow include ensuring they are atomic, meaning they either succeed or fail as a whole. Additionally, tasks should be idempotent, which means running the same task multiple times will yield the same result without causing any unexpected side effects. This allows for checkpoint features to resume from failed steps rather than re-executing them entirely.

Code examples are relevant:
```java
// Example of ensuring a task is atomic and idempotent in pseudocode
class Task {
    void execute() {
        // Execute logic that should be atomic and idempotent
    }

    boolean canRetryOnFailure() {
        return true; // Indicate if the task can be retried on failure
    }
}

// Example of a retry mechanism in pseudocode
void runTaskWithRetry(Task task) {
    int retries = 3;
    while (retries > 0) {
        try {
            task.execute();
            break; // Exit loop if execution succeeds
        } catch (Exception e) {
            retries--;
        }
    }
}
```
x??

---

#### Task Size in Computational DAGs
Background context: The size of tasks within a computational DAG should be balanced to facilitate debugging and traceability while avoiding unnecessary overhead. Tasks that are too small can introduce overhead, whereas overly large tasks may constrain debuggability.

:p What considerations should be taken into account when defining the size of tasks in a computational DAG?
??x
When defining the size of tasks within a computational DAG, consider balancing task size to facilitate debugging and traceability without introducing unnecessary overhead. Tasks that are too small can introduce overhead due to frequent execution and management costs. Conversely, overly large tasks may constrain debuggability and lead to costly retry operations.

Code examples are relevant:
```java
// Example of defining a balanced task in pseudocode
class Task {
    int size = 100; // Define an appropriate size for the task

    void execute() {
        // Execute logic that should be small enough to facilitate debugging but not too large
    }
}

void defineTaskSize(Task task) {
    if (task.size < MIN_SIZE || task.size > MAX_SIZE) {
        throw new IllegalArgumentException("Task size out of bounds");
    }
}
```
x??

---

---
#### Flexibility of WMS
Flexibility is a crucial property for a Workflow Management System (WMS) as it enables handling a wide variety of tasks and workflows. A flexible WMS should support both simple linear workflows and complex ones, with options to initiate workflows either through scheduling or events.

:p What does flexibility in WMS allow?
??x
A flexible WMS allows managing diverse tasks and workflows. It supports the creation of simple linear workflows as well as more complex ones, and offers flexibility in initiating workflows—both scheduled executions and event-triggered actions.
x??

---
#### Configurability of WMS
Configurability is another essential feature where users can define workflow specifications using Infrastructure-as-Code (IaC) methodologies. This allows defining workflows and tasks through programming languages like Python or Domain-Specific Languages (DSLs) such as JSON.

:p How does configurability enable workflow definition?
??x
Configurability enables users to programmatically define workflow and task specifications, including dependency structures, input/output parameters, error handling, timeout policies, retry logic, status updates, triggers, execution limits, and logging. For example, using a DSL like JSON or Python scripts, users can specify these details.
x??

---
#### Dependency Management in WMS
Dependency management is critical for WMS as it allows managing dependencies between tasks and workflows. More advanced WMSs support parallel workflow/task execution, dynamic task creation, and information passing.

:p How do more advanced WMSs manage dependencies?
??x
More advanced WMSs allow for parallel execution of workflows/tasks, dynamic task generation based on conditions, and the exchange of information between tasks. For instance, in Figure 11-4(a), tasks 3 and 4 are conditionally executed after task 2 based on a specific condition. In contrast, (b) illustrates a workflow that generates a variable number of tasks at runtime.
x??

---
#### Coordination Patterns in WMS
Coordination patterns in WMS describe how components like workflows and tasks interact with each other. Two notable patterns are the Synchronous Blocking Pattern (SBP) and the Asynchronous Non-Blocking Pattern (ANBP).

:p What are coordination patterns used for in WMS?
??x
Coordination patterns define interaction methods between workflows and tasks within a WMS. The Synchronous Blocking Pattern involves sequential task execution where each task waits for the previous one to complete, while the Asynchronous Non-Blocking Pattern allows running tasks independently and without waiting for responses.
x??

---
#### Example of SBP in WMS
The Synchronous Blocking Pattern (SBP) is used when components call each other and wait for a response. This pattern is typical in systems using HTTP and RESTful APIs.

:p Provide an example of the Synchronous Blocking Pattern in WMS?
??x
In an SBP, tasks within a workflow are executed sequentially within the same process, with each task waiting for the previous one to complete before proceeding. Here's a simple pseudocode example:
```pseudocode
function executeWorkflow(task1, task2, task3) {
    result1 = task1.execute()
    result2 = task2.execute(result1)
    result3 = task3.execute(result2)
}
```
x??

---
#### Example of ANBP in WMS
The Asynchronous Non-Blocking Pattern (ANBP) operates in a fire-and-forget mode, where components send requests or messages without waiting for responses. This can be implemented by running each workflow task independently and using a dedicated queue.

:p Provide an example of the Asynchronous Non-Blocking Pattern in WMS?
??x
In ANBP, tasks are executed independently on separate processes or machines with a dedicated task queue handling incoming messages. Here's a pseudocode example:
```pseudocode
function executeWorkflow(taskQueue) {
    task1.submitToQueue()
    task2.submitToQueue()
    // more tasks can be submitted to the queue
}
```
x??

---

---
#### Synchronous Blocking Pattern vs. Asynchronous Non-Blocking Pattern Scalability
The scalability of a Workflow Management System (WMS) is crucial for handling concurrent executions and tasks, especially in event-driven or high-volume systems. The ability to scale efficiently depends on how the WMS manages its workflows and tasks.
:p What are the key differences between synchronous blocking pattern and asynchronous non-blocking pattern in terms of scalability?
??x
The synchronous blocking pattern can become a bottleneck as it waits for each task to complete before proceeding, which limits the number of concurrent tasks that can be handled. In contrast, the asynchronous non-blocking pattern allows multiple tasks to run concurrently without waiting for others to finish, thus enhancing scalability.
```java
// Example of synchronous blocking pattern (pseudo-code)
public void processTasksSynchronously(List<Task> tasks) {
    for (Task task : tasks) {
        executeTask(task);
        // Wait for the task to complete before proceeding
    }
}
```
```java
// Example of asynchronous non-blocking pattern (pseudo-code)
public void processTasksAsynchronously(List<Task> tasks) {
    List<Future<TaskResult>> futures = new ArrayList<>();
    ExecutorService executor = Executors.newFixedThreadPool(tasks.size());
    for (Task task : tasks) {
        Future<TaskResult> future = executor.submit(task);
        futures.add(future);
    }
    // Process results without blocking
    for (Future<TaskResult> future : futures) {
        TaskResult result = future.get();
        process(result);
    }
}
```
x??

---
#### Scalability Requirements in WMSs
The scalability requirements of a WMS vary based on the nature and volume of tasks it handles. Event-driven systems, which deal with real-time events or continuous data streams, require more robust scalability to manage high volumes of concurrent tasks. In contrast, scheduled batch-oriented systems operate on predefined schedules and can process tasks in batches.
:p How do event-driven and high-volume WMSs typically differ from scheduled batch-oriented WMSs in terms of scalability requirements?
??x
Event-driven and high-volume WMSs require robust scalability capabilities to handle large volumes of tasks triggered by real-time events or continuous data streams. They need to efficiently manage a high number of concurrent executions without bottlenecks.
Scheduled batch-oriented WMSs, on the other hand, operate based on predefined schedules and process tasks in batches. Their scalability requirements are generally less stringent as they do not handle a large number of simultaneous tasks but focus on efficient processing during specific time intervals.
x??

---
#### Integration Capabilities in WMSs
A key feature of modern Workflow Management Systems is their seamless integration with other tools and technologies, particularly those that the WMS manages or coordinates. For example, cloud-based WMSs can integrate easily with cloud services running applications, making orchestration more convenient.
:p What makes a highly desired feature for WMSs in terms of integration?
??x
A highly desired feature for WMSs is their capability to seamlessly integrate with other tools and technologies, especially those that the WMS manages or coordinates. This includes being able to work smoothly with cloud services, databases, APIs, and other systems.
For instance, a cloud-based WMS can easily integrate with cloud services running applications, making orchestration more convenient as it leverages existing infrastructure without additional complexity.
x??

---

