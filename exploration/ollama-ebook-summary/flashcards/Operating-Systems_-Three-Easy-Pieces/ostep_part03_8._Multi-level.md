# Flashcards: Operating-Systems_-Three-Easy-Pieces_processed (Part 3)

**Starting Chapter:** 8. Multi-level Feedback

---

#### Multi-Level Feedback Queue (MLFQ) Overview
Background context explaining MLFQ. It is designed to address two primary challenges: optimizing turnaround time by prioritizing shorter jobs, and ensuring responsive behavior for interactive users by minimizing response time.

:p What are the main goals of MLFQ in scheduling?
??x
The main goals of MLFQ are to optimize turnaround time by running shorter jobs first and to ensure a responsive system for interactive users by reducing their waiting times. These objectives are challenging because traditional algorithms like Shortest Job First (SJF) or Round Robin (RR) excel at one but struggle with the other due to limitations in predicting job duration.
x??

---

#### Priority Levels in MLFQ
Explanation of how MLFQ uses multiple priority levels, each corresponding to a different queue.

:p How does MLFQ handle prioritization among ready jobs?
??x
MLFQ assigns distinct queues (priority levels) to different processes based on their priorities. At any given time, a job that is ready to run is placed in one of these queues. Higher priority queues are checked first for running jobs. If multiple jobs have the same priority level, MLFQ uses round-robin scheduling among them.
x??

---

#### Round-Robin Scheduling
Explanation and example of how round-robin scheduling works within MLFQ.

:p How does round-robin scheduling operate in MLFQ?
??x
In MLFQ, if multiple jobs have the same priority level, a round-robin approach is used to select which job runs next. This means that each job gets an equal amount of CPU time before moving on to the next one in the queue.

Code example:
```java
public class RoundRobinScheduling {
    private int quantum = 10; // Time slice for each process

    public void scheduleJobs(List<Process> jobs) {
        while (!jobs.isEmpty()) {
            Process currentJob = jobs.remove(0);
            if (currentJob.runTime < quantum) {
                // Job completes within the time slice
                System.out.println("Executing: " + currentJob.id);
            } else {
                // Job needs more time, add it back to the end of the queue
                currentJob.runTime -= quantum;
                jobs.add(currentJob);
            }
        }
    }
}
```
x??

---

#### Learning from History in Scheduling
Explanation on how MLFQ and similar systems use historical data to improve future decisions.

:p Why is learning from history important in scheduling?
??x
Learning from history is crucial because it allows the scheduler to adapt its behavior based on the actual performance of jobs. By observing patterns and behaviors over time, the scheduler can make more informed decisions about which processes to prioritize or how much CPU time to allocate. However, this approach requires careful implementation to avoid making worse decisions than those made without historical data.
x??

---

#### Practical Considerations
Explanation on practical challenges and considerations in implementing MLFQ.

:p What are some practical challenges in implementing MLFQ?
??x
Implementing MLFQ involves several practical challenges:
1. **Queue Management**: Efficiently managing multiple queues with varying priority levels.
2. **Scheduling Algorithms**: Choosing appropriate algorithms for different phases of job execution (e.g., round-robin within the same queue).
3. **System Overhead**: Managing the overhead introduced by additional scheduling and queue management logic.
4. **Dynamic Adjustment**: Dynamically adjusting priorities based on real-time system behavior without causing excessive oscillations.

These challenges require careful design to ensure that MLFQ provides both low response times for interactive jobs and good turnaround time for longer-running tasks.
x??

---

#### MLFQ Scheduling Overview
Background context: The Multi-Level Feedback Queue (MLFQ) is a scheduling algorithm that aims to balance between short-running interactive jobs and long-running CPU-bound tasks. It uses multiple queues with different priorities, allowing for dynamic job prioritization based on observed behavior.

:p What are the two basic rules of MLFQ?
??x
Rule 1: If Priority(A) > Priority(B), A runs (B doesn’t).
Rule 2: If Priority(A) = Priority(B), A & B run in Round Robin.
x??

---
#### Rule-Based Scheduling in MLFQ
Background context: The rules of MLFQ help to determine which job gets the CPU at any given time. The key is dynamically adjusting priorities based on a job's behavior.

:p How do these rules work?
??x
- **Rule 1**: If Job A has a higher priority than Job B, it will run while Job B doesn't.
- **Rule 2**: If both jobs have the same priority, they share CPU time in Round Robin fashion.
x??

---
#### Priority Adjustment Algorithm
Background context: The algorithm for adjusting job priorities over time is crucial to MLFQ. It aims to adapt based on a job's behavior, placing more interactive tasks higher and longer-running tasks lower.

:p How does the initial placement of jobs work?
??x
- **Rule 3**: When a job enters the system, it starts at the highest priority (topmost queue).
x??

---
#### Priority Adjustment Algorithm: Time Slice Rules
Background context: The rules for changing priorities involve time slices. Jobs that use up their full slice lose priority, while those that relinquish early stay in place.

:p What happens if a job uses up an entire timeslice?
??x
- **Rule 4a**: If a job uses up its entire timeslice without yielding the CPU, its priority is reduced (it moves down one queue).
x??

---
#### Priority Adjustment Algorithm: Time Slice Rules (Continued)
Background context: The rules for changing priorities involve time slices. Jobs that use up their full slice lose priority, while those that relinquish early stay in place.

:p What happens if a job gives up the CPU before its timeslice is over?
??x
- **Rule 4b**: If a job relinquishes the CPU before the timeslice ends, it stays at the same priority level.
x??

---
#### Example of Long-Running Job Behavior
Background context: An example helps to understand how MLFQ handles long-running jobs. These jobs typically start high and move lower as they use up their timeslices.

:p How does a single long-running job behave in a three-queue MLFQ scheduler?
??x
- The job enters at the highest priority (Q2).
- After one time-slice of 10 ms, its priority is reduced to Q1.
- After running on Q1 for another timeslice, it gets lowered to the lowest priority queue (Q0) and stays there.

Example:
```plaintext
Time: 0 - 10ms: Job at Q2 -> Priority decreased to Q1 after one full timeslice.
Time: 10 - 20ms: Job at Q1 -> Another timeslice, no change in priority.
Time: 20 - 30ms: Job at Q0 -> Final timeslice, job stays in Q0.
```
x??

---
#### Priority Levels and Queues
Background context: The number of queues and their priorities are key to understanding MLFQ. Typically, there are multiple queues with different priority levels.

:p How many queue levels are mentioned in the text?
??x
- There are at least three queue levels (Q0, Q1, Q2) as described.
x??

---
#### Queue Visualization Example
Background context: A snapshot of queues shows how MLFQ manages jobs. In such a snapshot, certain jobs may be prioritized higher than others.

:p How does the example with two high-priority jobs and one low-priority job work?
??x
- Two jobs (A and B) are at the highest priority level.
- Job C is in the middle, and Job D is at the lowest priority. 
- The scheduler alternates time slices between A and B because they are the highest priority jobs; jobs C and D never get a chance to run.

Example:
```plaintext
Q1 (High Priority): A & B
Q2: C
Q3 (Low Priority): D
```
x??

---

#### MLFQ and SJF Approximation
Background context: The Multi-Level Feedback Queue (MLFQ) tries to approximate Shortest Job First (SJF) by initially giving high priority to short jobs, which can complete quickly. This helps in running interactive jobs efficiently while allowing long-running batch jobs to eventually get CPU time.

:p How does MLFQ attempt to approximate SJF?
??x
MLFQ approximates SJF by initially assigning higher priority to new jobs, assuming they might be short and need quick completion. If a job turns out to be long-running, it will eventually move down the queue levels and thus get its fair share of CPU time.

```java
// Pseudocode for MLFQ priority assignment based on job type
public void assignPriority(Job job) {
    if (job.isInteractive()) { // Check if job is interactive
        job.setPriority(highPriority);
    } else if (job.isBatch()) { // Otherwise, check if batch
        job.setPriority(lowPriority);
    }
}
```
x??

---

#### Priority Boost for Interactive Jobs
Background context: MLFQ keeps interactive jobs at a higher priority level even when they relinquish the CPU before their time slice ends. This is to ensure that interactive jobs are given quick responses, as they might need frequent CPU access.

:p How does MLFQ handle I/O-intensive jobs?
??x
MLFQ ensures that I/O-intensive jobs (interactive jobs) remain at a high priority level even when they release the CPU before their time slice ends. This is achieved by not penalizing such jobs for relinquishing the CPU, which helps in maintaining quick response times.

```java
// Pseudocode for handling I/O jobs in MLFQ
public void handleIOJob(Job job) {
    if (job.isIOHeavy()) { // Check if job is I/O heavy
        // Keep at current priority
    } else {
        // Process the job normally
    }
}
```
x??

---

#### Starvation Problem in MLFQ
Background context: The MLFQ algorithm faces a significant issue known as starvation. If too many interactive jobs are present, they might consume all CPU time, leaving long-running batch jobs without any execution.

:p What is the main problem with the current MLFQ implementation?
??x
The main problem with the current MLFQ implementation is that it can lead to starvation of long-running batch jobs if there are too many interactive jobs. These interactive jobs might consume all available CPU time, leaving no opportunity for long-running jobs to execute.

```java
// Pseudocode for detecting and handling starvation in MLFQ
public void checkForStarvation() {
    int interactiveJobs = countInteractiveJobs();
    if (interactiveJobs > threshold) { // Check if too many interactive jobs
        log("Potential starvation detected. Long-running jobs might starve.");
    }
}
```
x??

#### Gaming the Scheduler
Background context: The text discusses how a smart user can exploit the current scheduling algorithm to gain more CPU time by issuing an I/O operation before their time slice is over, thereby relinquishing the CPU and remaining in the same queue. This can allow them to monopolize the CPU when done correctly.
:p What does gaming the scheduler refer to?
??x
Gaming the scheduler refers to a technique where a user manipulates their program to gain more than its fair share of resources by issuing an I/O operation before their time slice is over, thereby relinquishing the CPU and remaining in the same queue. This allows them to monopolize the CPU if done correctly.
x??

---

#### Problem with Current Scheduling
Background context: The current scheduling algorithm can lead to starvation, where long-running jobs do not get sufficient CPU time due to shorter, more interactive jobs continuously using the CPU.
:p What is the main problem with the current scheduling approach?
??x
The main problem with the current scheduling approach is that it can lead to starvation. Long-running jobs may not receive enough CPU time because they are often preempted by short, interactive jobs, which can keep them waiting indefinitely for their turn on the CPU.
x??

---

#### Priority Boost Rule
Background context: To address the issue of long-running jobs starving and interactive jobs being properly handled, a new rule is introduced to periodically boost the priority of all jobs. This ensures that even if a job becomes more interactive over time, it will still be treated appropriately by the scheduler.
:p How does the priority boost rule solve the problem of starvation?
??x
The priority boost rule solves the problem of starvation by periodically moving all jobs in the system to the topmost queue after a certain time period S. This ensures that long-running CPU-bound jobs get some CPU time, and if they become more interactive, they are treated properly as well.
x??

---

#### Behavior of Priority Boost
Background context: The priority boost rule is illustrated through an example where a long-running job competes with two short-running interactive jobs. Without the priority boost, the long-running job gets starved. With the priority boost every 50 ms, the long-running job makes some progress.
:p How does the behavior of the priority boost affect the long-running job?
??x
The behavior of the priority boost affects the long-running job by periodically moving it to the topmost queue after a certain time period S (every 50 ms in this example). This ensures that even if the job becomes more interactive, it still receives some CPU time, preventing starvation and ensuring proper treatment.
x??

---

#### Code Example for Priority Boost
Background context: The priority boost rule can be implemented by periodically moving all jobs to the top queue. Below is a simple implementation of this concept in pseudocode.
:p Show an example of how to implement the priority boost in code.
??x
```java
// Pseudocode for implementing the priority boost

public class Scheduler {
    private int timePeriodS; // Time period after which to boost priorities
    
    public void scheduleJobs(List<Job> jobs) {
        long currentTime = System.currentTimeMillis();
        
        if (currentTime - lastBoostTime > timePeriodS) {
            // Move all jobs to the topmost queue
            for (Job job : jobs) {
                moveJobToTop(job);
            }
            lastBoostTime = currentTime;
        }
    }

    private void moveJobToTop(Job job) {
        // Logic to move a job to the top of the queue
    }
}
```
x??

---

#### Descriptions Differentiation
- **Gaming the Scheduler**: Focuses on user manipulation techniques.
- **Problem with Current Scheduling**: Highlights starvation and interactive job handling issues.
- **Priority Boost Rule**: Introduces periodic priority boosts for all jobs.
- **Behavior of Priority Boost**: Illustrates how it affects long-running jobs.
- **Code Example for Priority Boost**: Provides a code implementation detail.

#### Time Slice Scheduling Considerations
Background context: The document discusses the challenge of setting the time slice (S) parameter for scheduling algorithms, particularly in the context of the Multi-Level Feedback Queue (MLFQ). If set too high, long-running jobs could starve; if set too low, interactive jobs may not get enough CPU time.
:p What is the primary concern with setting the time slice (S) in a Multi-Level Feedback Queue?
??x
The primary concern with setting the time slice (S) in a Multi-Level Feedback Queue is finding an optimal value that prevents long-running jobs from starving while ensuring sufficient CPU time for interactive jobs. If S is too high, it can lead to inefficiencies where longer processes dominate the CPU, potentially delaying other tasks. Conversely, if S is too low, shorter, more frequent context switches could degrade system performance.
x??

---
#### Voo-Doo Constants and MLFQ
Background context: John Ousterhout referred to certain parameters in systems as "voo-doo constants" because their correct values seemed to require some form of black magic. In the case of the Multi-Level Feedback Queue (MLFQ), setting the time slice S correctly is challenging.
:p What term did John Ousterhout use to describe parameters like the time slice (S) in systems?
??x
John Ousterhout used the term "voo-doo constants" to describe parameters in systems, such as the time slice (S) in MLFQ, because their correct values seemed to require some form of black magic or complex, seemingly arbitrary determination.
x??

---
#### Addressing Gaming with New Rules
Background context: The text mentions that rules 4a and 4b allowed jobs to retain their priority by relinquishing the CPU before the time slice expired. To prevent gaming of the scheduler, a new rule was implemented to ensure better accounting of CPU usage at each level.
:p What change was made to prevent gaming in the MLFQ scheduler?
??x
A change was made to prevent gaming in the MLFQ scheduler by rewriting Rule 4 as follows: once a job uses up its time allotment at a given level, regardless of how many times it has relinquished the CPU, its priority is reduced (i.e., it moves down one queue). This ensures that jobs cannot retain their high priority indefinitely just by yielding control before the time slice expires.
x??

---
#### Tuning MLFQ: Parameterization
Background context: The document discusses the challenges of tuning a Multi-Level Feedback Queue (MLFQ) scheduler, including how many queues to use, time slice size per queue, and frequency of priority boosting. There are no easy answers to these questions, and experience with workloads is necessary for optimal parameterization.
:p What issues arise when tuning a MLFQ scheduler?
??x
When tuning a Multi-Level Feedback Queue (MLFQ) scheduler, several key issues arise:
- How many queues should be used?
- How big should the time slice be per queue?
- How often should priority be boosted to avoid starvation and account for changes in behavior?

There are no easy answers to these questions. Experience with workloads and subsequent tuning of the scheduler will help achieve a satisfactory balance.
x??

---

#### Ousterhout's Law on MLFQ Variants
Background context explaining the concept. The Multilevel Feedback Queue (MLFQ) scheduling algorithm allows for varying time-slice lengths across different queues, optimizing performance by giving short time slices to high-priority interactive jobs and longer time slices to low-priority CPU-bound jobs.
:p What is Ousterhout's Law in the context of MLFQ?
??x
Ousterhout's Law states that most MLFQ variants allow for varying time-slice lengths across different queues. High-priority queues are given short time slices (e.g., 10-20 milliseconds) to handle interactive jobs, while low-priority queues with longer-running CPU-bound tasks receive longer time slices (e.g., 100+ milliseconds).
??x
---

#### Solaris MLFQ Implementation: Time-Sharing Scheduling Class (TS)
Background context explaining the concept. The Time-Sharing scheduling class in Solaris provides a configurable framework for managing priorities and time-slices across multiple queues.
:p How does the Solaris Time-Sharing scheduling class manage process priorities?
??x
The Solaris Time-Sharing scheduling class uses tables to define how process priorities change over their lifetimes, how long each time slice is, and how often job priorities are boosted. The default configuration includes 60 queues with increasing time-slice lengths from 20 milliseconds (highest priority) to a few hundred milliseconds (lowest), and priorities are typically boosted every second.
??x
---

#### Other MLFQ Schedulers: Formula-Based Priority Adjustment
Background context explaining the concept. Some MLFQ schedulers, like FreeBSD's version 4.3 scheduler, use mathematical formulas to calculate job priorities based on CPU usage. Priorities can decay over time, providing a different way of boosting priorities.
:p How do other MLFQ schedulers adjust process priorities differently from Solaris TS?
??x
Other MLFQ schedulers adjust priorities using mathematical formulas instead of tables and rules described in the text. For example, FreeBSD's version 4.3 scheduler calculates current priority levels based on CPU usage and decays this usage over time to provide a different form of priority boost.
??x
---

#### Additional Features: Reserved Priority Levels and User Advice
Background context explaining the concept. Some schedulers reserve certain priority levels for system processes, while others allow user advice through commands like `nice` to influence scheduling decisions.
:p What are some additional features found in many schedulers?
??x
Additional features include reserving the highest priority levels for operating system work, meaning typical user jobs cannot obtain these highest priorities. Some systems also provide user advice through command-line utilities like `nice`, allowing users to adjust job priorities and thus influence their chances of running at any given time.
??x
---

#### MLFQ Overview
MLFQ stands for Multi-Level Feedback Queue. It is a scheduling approach used to manage jobs of varying characteristics, such as short-running interactive tasks and long-running CPU-intensive workloads.

:p What is MLFQ and why is it named so?
??x
MLFQ is a scheduling mechanism that uses multiple levels of queues. The term "feedback" comes from the fact that the system observes how jobs behave over time and adjusts their priorities accordingly to achieve better performance.
x??

---
#### Rule 1: Priority-based Scheduling
This rule states that if the priority of process A is higher than that of process B, then A will run instead of B.

:p According to Rule 1, what determines whether a process runs?
??x
According to Rule 1, a process runs based on its priority. If Process A has a higher priority (Priority(A) > Priority(B)), it gets scheduled and executed over Process B.
x??

---
#### Rule 2: Round-Robin Scheduling for Equal Priorities
When two processes have the same priority, they are scheduled using round-robin with a predefined time slice.

:p How do you handle scheduling when multiple processes have equal priorities?
??x
When multiple processes share the same priority (Priority(A) = Priority(B)), Rule 2 dictates that these processes should be scheduled in a round-robin fashion. Each process gets a quantum length of CPU time, and then control passes to the next process with the same priority.
x??

---
#### Rule 3: Initial Job Placement
When a new job enters the system, it is initially placed at the highest priority level (the topmost queue).

:p Where does a newly arrived job start in MLFQ?
??x
A newly arrived job starts its lifecycle by being placed in the highest-priority queue. This ensures that short-running interactive jobs can get immediate attention and better performance.
x??

---
#### Rule 4: Priority Reduction on Time Expiry
If a job uses up its time allotment at a given priority level, its priority is reduced (it moves down to the next lower priority queue).

:p What happens when a job's time slice expires?
??x
When a job's allocated time slice expires regardless of how many times it has voluntarily given up the CPU, according to Rule 4, its priority is decreased. This means the job will move to the next lower-priority queue in the MLFQ.
x??

---
#### Rule 5: Periodic Queue Reordering
Every period S, all jobs in the system are moved back to the topmost queue.

:p How does MLFQ manage periodic re-evaluation of jobs?
??x
Rule 5 states that every period S (a defined time interval), all processes in the system are moved back to the highest-priority queue. This ensures that the scheduler periodically reassesses the priorities and execution needs of the processes.
x??

---
#### Performance Characteristics of MLFQ
MLFQ is designed to provide excellent performance for short-running interactive jobs, while ensuring fair treatment and progress for long-running CPU-intensive workloads.

:p How does MLFQ balance between interactive and long-running tasks?
??x
MLFQ achieves a balanced approach by dynamically adjusting priorities based on observed job behavior. For short-running interactive jobs, it ensures quick response times through higher initial priority levels. For long-running CPU-intensive tasks, it allows them to progress over time while maintaining fairness.
x??

---
#### Historical Context and Usage
Many operating systems, including BSD UNIX derivatives, Solaris, and Windows NT, use variations of MLFQ as their base scheduler.

:p In which operating systems is MLFQ commonly implemented?
??x
MLFQ is widely used in various modern operating systems. It can be found in systems like BSD UNIX derivatives [LM+89, B86], Solaris [M06], and Windows NT and subsequent versions of the Windows operating system [CS97].
x??

---

#### 4.3BSD Unix Operating System Book
This book, "The Design and Implementation of the 4.3BSD UNIX Operating System," is a classic written by four key contributors to BSD. It provides insights into the design and implementation of an early version of the UNIX operating system.

:p What does this flashcard cover?
??x
This flashcard covers the historical significance of the "The Design and Implementation of the 4.3BSD UNIX Operating System" book, highlighting its importance as a reference for understanding the architecture and design principles of the 4.3BSD version of UNIX.
x??

---

#### Solaris Internals Book
Richard McDougall's book, "Solaris Internals: Solaris 10 and OpenSolaris Kernel Architecture," delves into the workings of Solaris operating system.

:p What is this flashcard about?
??x
This flashcard describes Richard McDougall's book that focuses on the internal architecture and working mechanisms of Solaris OS, particularly versions 10 and OpenSolaris. It serves as a comprehensive guide for understanding how Solaris operates at a deep level.
x??

---

#### John Ousterhout’s Home Page
John Ousterhout's home page offers valuable resources and insights into his academic work.

:p What does this flashcard discuss?
??x
This flashcard refers to the home page of Professor John Ousterhout, which contains various resources related to his teaching and research. It mentions that one of the co-authors of a book had the opportunity to study under him in graduate school, leading to personal connections such as marriage and collaboration.
x??

---

#### Informed Prefetching and Caching Paper
The paper "Informed Prefetching and Caching" discusses innovative ideas for file systems.

:p What is this flashcard for?
??x
This flashcard introduces the paper "Informed Prefetching and Caching," which explores advanced concepts in file systems, including how applications can provide guidance to the operating system about their future data access patterns. This knowledge helps in optimizing I/O operations.
x??

---

#### Scheduling Workload Analysis Paper
The paper discusses challenges in scheduling within distributed storage systems.

:p What is covered in this flashcard?
??x
This flashcard highlights a recent work that examines the complexities of scheduling input/output (I/O) requests in modern distributed storage systems such as Hive/HDFS, Cassandra, MongoDB, and Riak. The study emphasizes the potential for single users to monopolize system resources without proper management.
x??

---

#### MLFQ Scheduler Simulation
The `mlfq.py` program allows you to experiment with the Multi-Level Feedback Queue (MLFQ) scheduler.

:p What is this flashcard about?
??x
This flashcard introduces a Python simulation, `mlfq.py`, that demonstrates how the Multi-Level Feedback Queue (MLFQ) scheduler functions. It provides exercises for understanding and experimenting with different aspects of the MLFQ scheduler, such as configuration parameters and behavior patterns.
x??

---

#### MLFQ Scheduler Simulation Questions

1. **Run a few randomly-generated problems with just two jobs and two queues; compute the MLFQ execution trace for each.**
:p How would you approach this question?
??x
To address this, generate random job scenarios involving two jobs and two MLFQ queues. For each scenario, run the simulation to observe how jobs are scheduled across different priority levels and record the execution trace.

Example:
```python
# Pseudocode for generating a simple test case
def simulate_two_jobs_two_queues():
    # Initialize jobs and their priorities
    job1 = {'priority': 2, 'execution_time': 5}
    job2 = {'priority': 3, 'execution_time': 7}
    
    # Simulate the MLFQ scheduler behavior
    execution_trace = run_mlfaq(job1, job2)
    print(execution_trace)

# Function to simulate MLFQ behavior (simplified for illustration)
def run_mlfaq(job1, job2):
    trace = []
    current_time = 0
    
    # Simulate quantum usage and priority changes
    while job1['execution_time'] > 0 or job2['execution_time'] > 0:
        if job1['priority'] == current_priority_level():
            trace.append("Executing Job 1")
            job1['execution_time'] -= 1
        elif job2['priority'] == current_priority_level():
            trace.append("Executing Job 2")
            job2['execution_time'] -= 1
    
    return trace

def current_priority_level():
    # Determine the priority level based on predefined logic
    pass
```
x??

---

#### Configuring MLFQ Scheduler Parameters for Round-Robin Behavior

:p How would you configure the scheduler to behave like a round-robin scheduler?
??x
To configure the MLFQ scheduler so that it behaves like a round-robin scheduler, set all priority levels' quantum lengths to be equal and minimal. This ensures that each job gets an equal amount of CPU time.

Example:
```python
# Configuration for round-robin behavior in MLFQ
def configure_round_robin():
    # Set the same small quantum length for all priorities
    quantum_lengths = [10] * 4
    
    # Set other parameters as needed, but ensure no priority level dominates
    scheduling_params['quantum_lengths'] = quantum_lengths

configure_round_robin()
```
x??

---

#### Gaming the Scheduler with Rules 4a and 4b

:p How can you craft a workload to game the MLFQ scheduler using Rules 4a and 4b?
??x
To exploit Rules 4a and 4b (which typically involve prioritizing certain jobs based on their historical behavior), create a workload where one job always requests high-priority tasks, forcing it to run frequently. This can be achieved by simulating a scenario where the first job constantly issues requests that are prioritized under Rule 4a or 4b.

Example:
```python
# Simulate a job that gamingly exploits rules 4a and 4b
def simulate_gaming_workload():
    # Create two jobs with different priorities
    job1 = {'priority': 2, 'execution_time': 5}
    job2 = {'priority': 3, 'execution_time': 7}
    
    # Run the simulation to observe the gaming effect
    execution_trace = run_mlfaq(job1, job2)
    print(execution_trace)

def run_mlfaq(job1, job2):
    trace = []
    current_time = 0
    
    while job1['execution_time'] > 0 or job2['execution_time'] > 0:
        if job1['priority'] == current_priority_level() and is_gaming_condition_met():
            trace.append("Executing Job 1")
            job1['execution_time'] -= 1
        elif job2['priority'] == current_priority_level():
            trace.append("Executing Job 2")
            job2['execution_time'] -= 1
    
    return trace

def is_gaming_condition_met():
    # Simulate the gaming condition for job1
    pass

def current_priority_level():
    # Determine the priority level based on predefined logic and gaming conditions
    pass
```
x??

---

#### Scheduling with Quantum Length in Highest Queue

:p How often would you have to boost jobs back to the highest priority level?
??x
To ensure a long-running job gets at least 5 percent of the CPU, given a quantum length of 10 ms in the highest queue, calculate how frequently you need to boost the job back. If the job runs for 95% of the time without being boosted and only 5% when boosted:

Example:
```python
# Calculate frequency of boosting jobs
def boost_frequency():
    # Assume a quantum length of 10 ms in the highest queue
    quantum_length = 10
    
    # Desired CPU percentage
    desired_cpu_percentage = 5 / 100.0
    
    # Frequency to ensure at least 5% CPU time
    frequency = (desired_cpu_percentage * 1000) / quantum_length
    
    print(f"Boost the job back every {frequency} ms")
    
boost_frequency()
```
x??

---

#### I/O Completion Handling

:p How does the -I flag affect the MLFQ scheduler simulation?
??x
The `-I` flag in this scheduling simulator determines where to add a job that just finished I/O operations. Experimenting with different values of the `-I` flag can help understand how it influences the placement and execution sequence of jobs.

Example:
```python
# Simulate different behaviors based on -I flag value
def test_io_completion_handling():
    # Use different -I flag values to observe changes in job scheduling
    io_completion_flags = [0, 1, 2]
    
    for flag in io_completion_flags:
        print(f"Testing with -I{flag} flag")
        simulation_result = run_mlfaq(flag)
        print(simulation_result)

def run_mlfaq(flag):
    # Simulate the MLFQ behavior based on the given -I flag
    pass

test_io_completion_handling()
```
x??

