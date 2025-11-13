# Flashcards: Operating-Systems_-Three-Easy-Pieces_processed (Part 27)

**Starting Chapter:** 8. Multi-level Feedback

---

#### Multi-Level Feedback Queue (MLFQ) Overview
Background context: The MLFQ is designed to optimize both turnaround time and response time, which are critical for different types of processes. MLFQ achieves this by dividing jobs into different priority levels and scheduling them based on their priorities. Each queue has a specific role in managing the job's execution.
:p What is the primary goal of implementing an MLFQ scheduler?
??x
The primary goal of implementing an MLFQ scheduler is to optimize both turnaround time (for shorter jobs) and response time (for interactive processes). This is achieved by dividing jobs into different priority levels, allowing the system to handle short-term and long-term scheduling needs more effectively.
x??

---

#### Priority Levels in MLFQ
Background context: In an MLFQ scheduler, each job belongs to a specific queue based on its priority. The queues are arranged in such a way that higher-priority jobs are given preference over lower-priority ones. This hierarchical structure helps the system balance between optimizing response time and turnaround time.
:p How does the MLFQ determine which job gets executed at any given time?
??x
The MLFQ determines which job gets executed at any given time by assessing the priority of each queue. Jobs on higher-priority queues have a better chance of being selected for execution. If multiple jobs are present in a queue with the same priority, round-robin scheduling is used to choose among them.
x??

---

#### Round-Robin Scheduling
Background context: When multiple jobs share the same priority level, the scheduler uses round-robin scheduling to fairly distribute CPU time among these processes. This ensures that no single job monopolizes the CPU, which could potentially starve other processes waiting for their turn.
:p How does round-robin scheduling work in MLFQ?
??x
Round-robin scheduling works by giving each job a fixed time slice (quantum) on the CPU. Once a process uses up its quantum, control is passed to the next job in the queue. This ensures that even lower-priority jobs get some CPU time, preventing starvation.
x??

---

#### Job Characteristics and Learning
Background context: The MLFQ scheduler aims to adaptively learn about the characteristics of running processes as it runs. By observing the behavior of jobs over time, the scheduler can make better decisions in future scheduling cycles. This adaptive learning helps improve both response times for interactive users and turnaround times for batch jobs.
:p Why is adaptive learning important in an MLFQ scheduler?
??x
Adaptive learning is crucial in an MLFQ scheduler because it allows the system to understand the nature of running processes over time. By observing job behavior, the scheduler can refine its scheduling decisions, leading to better overall performance and resource utilization.
x??

---

#### Historical Context of MLFQ
Background context: The concept of MLFQ was first introduced in 1962 with the Compatible Time-Sharing System (CTSS), and later refined in systems like Multics. This historical development highlights how operating system scheduling algorithms have evolved to balance between different types of workloads.
:p What led to the creation of the Multi-Level Feedback Queue scheduler?
??x
The Multi-Level Feedback Queue (MLFQ) scheduler was created to address the need for balancing response times for interactive users and turnaround times for batch jobs. It evolved from early systems like CTSS, where initial scheduling algorithms had limitations in handling diverse workloads effectively.
x??

---

#### Summary of Key Points
Background context: The MLFQ scheduler is designed to tackle the dual goals of optimizing both turnaround time and response time by using a multi-level queue system with round-robin scheduling. This approach allows for adaptive learning based on observed job behavior, making it more effective in dynamic workloads.
:p What are the key takeaways from this section about MLFQ?
??x
Key takeaways include:
- MLFQ optimizes both turnaround time and response time by using multiple priority queues.
- Jobs with higher priorities get executed first through round-robin scheduling within queues of equal priority.
- The scheduler learns from job behavior over time to make better scheduling decisions.
x??

---

#### MLFQ Overview
Background context explaining the Multi-Level Feedback Queue (MLFQ) scheduling algorithm. MLFQ varies job priorities based on observed behavior to improve system responsiveness and resource allocation efficiency.

:p What is MLFQ, and how does it work?
??x
MLFQ is a scheduling algorithm that uses multiple queues with different priority levels to manage processes more effectively. The key idea is that the scheduler adjusts the priority of jobs based on their behavior. For instance, if a job frequently relinquishes the CPU while waiting for keyboard input, its priority will remain high as it behaves like an interactive process. Conversely, if a job uses the CPU intensively for long periods, its priority will be reduced to allow more time for other processes.

The algorithm works by dynamically adjusting priorities rather than assigning fixed ones. This allows MLFQ to adapt to various types of jobs in a system, such as short-running interactive tasks and longer-running CPU-bound tasks.
x??

---

#### Priority Adjustment Rules
Background context on the rules that govern how MLFQ adjusts job priorities.

:p What are the priority adjustment rules for MLFQ?
??x
MLFQ uses the following priority adjustment rules:
1. **Rule 3**: When a job enters the system, it is placed at the highest priority (topmost queue).
2. **Rule 4a**: If a job uses up an entire time slice while running, its priority is reduced by one level.
3. **Rule 4b**: If a job gives up the CPU before the time slice is up, its priority remains unchanged.

These rules help ensure that short-running interactive jobs stay in higher priority queues and receive timely service, while longer-running processes are moved to lower priority queues.
x??

---

#### Example of Long-Running Job
Background context on how long-running jobs are handled by MLFQ.

:p How does MLFQ handle a long-running job?
??x
A long-running job is initially assigned the highest priority (topmost queue). If it runs for a full time slice without giving up control, its priority is reduced. This process continues until the job reaches the lowest priority queue where it remains unless there are changes in behavior.

For example, consider a three-queue MLFQ scheduler:
```plaintext
Q2  Q1  Q0
```
A long-running job would start at `Q2` and move down to `Q1`, then finally to `Q0`.

```java
// Pseudocode for priority adjustment
public void adjustPriority(int currentQueue, int timeSlice) {
    if (timeSlice == fullTimeSlice && !jobGaveUp()) {
        // Move job one level down in the queue
        currentQueue--;
    }
}
```
x??

---

#### MLFQ Overview and SJF Approximation
Background context: The Multi-Level Feedback Queue (MLFQ) algorithm aims to approximate Shortest Job First (SJF) scheduling while managing different types of jobs, such as long-running CPU-intensive tasks and short-running interactive tasks. MLFQ works by dividing the system into multiple priority queues where each queue processes jobs with a certain level of urgency.

:p What is MLFQ trying to achieve?
??x
MLFQ tries to approximate SJF scheduling by managing different types of jobs in varying priority levels, ensuring that short-running interactive jobs get executed quickly while long-running CPU-intensive jobs are scheduled appropriately.
x??

---

#### Arrival and Scheduling of Short Jobs
Background context: In the scenario provided, a long-running job (A) is already running when a short-running job (B) arrives. The MLFQ algorithm tries to prioritize B by inserting it into the highest-priority queue due to its expected shorter execution time.

:p What happens if a short job arrives during the execution of another long job?
??x
If a short job (e.g., B) arrives when a long-running job (A) is executing, MLFQ will insert the short job into the highest priority queue. This allows the system to potentially switch to the short job quickly and complete it before B moves down through lower priority queues.

:p How does MLFQ handle the insertion of a short job?
??x
MLFQ inserts a newly arrived short job (B) into the highest-priority queue, assuming that it might be completed within one or two time slices. If this assumption is correct and B is indeed short, it will complete quickly; otherwise, if B turns out to be long-running, it will move down through lower priority queues.
x??

---

#### Priority Levels in MLFQ
Background context: The example shows how jobs are distributed across different priority levels (Q0, Q1, Q2). Each level processes jobs with varying urgency. A job like B, which is short and interactive, starts at the highest priority but moves down if it takes longer than expected.

:p How do jobs move between priority queues in MLFQ?
??x
Jobs start in higher-priority queues (e.g., Q0) and may be moved to lower-priority queues based on their execution time. If a job is short, like B, it will complete quickly and stay at high priority; if long-running, like A, it will move down through the queues over time.

:p What determines the movement of jobs between queues?
??x
The movement of jobs between queues depends on the completion time of the job. Short jobs tend to complete within a few time slices and remain at higher priorities. Long-running jobs take longer to execute and thus move down through lower-priority queues.
x??

---

#### Handling I/O-Intensive Jobs in MLFQ
Background context: The example demonstrates how an interactive job (B) with high I/O requirements can be managed by MLFQ. According to Rule 4b, if a process gives up the processor before using up its time slice due to I/O operations, it remains at the same priority level.

:p How does MLFQ handle jobs that frequently release the CPU for I/O?
??x
MLFQ keeps interactive jobs like B at the highest priority because they often release the CPU during their execution. This ensures that such short, interactive jobs are scheduled quickly and efficiently without being penalized.

:p What rule in MLFQ dictates this behavior?
??x
Rule 4b of MLFQ states that if a process releases the CPU before completing its time slice due to I/O operations, it remains at the same priority level. This ensures that short or interactive jobs are not penalized for their frequent use of I/O.
x??

---

#### Starvation in MLFQ
Background context: The example highlights a potential flaw in MLFQ where too many interactive jobs can consume all CPU time, starving long-running jobs.

:p What is the primary risk associated with having too many interactive jobs in MLFQ?
??x
The primary risk of having too many interactive jobs in MLFQ is that they might monopolize the CPU, preventing long-running tasks from getting any execution time and leading to starvation.

:p How can this issue be mitigated in MLFQ?
??x
To mitigate the issue of starvation, MLFQ must ensure a balanced distribution of CPU time among different types of jobs. Techniques such as priority inversion or adjusting time slices might help prevent long-running tasks from being starved.
x??

---

#### Gaming the Scheduler Attack
Background context: The provided text discusses a security issue where users can "game" the scheduler by issuing I/O operations to relinquish CPU time temporarily, thereby gaining more CPU time than their fair share. This is particularly problematic in scenarios like modern datacenters where multiple users share resources.

:p What is an example of gaming the scheduler?
??x
A smart user could write a program that frequently issues an I/O operation just before its time slice ends to relinquish control of the CPU, thereby remaining in the same priority queue and gaining more CPU time than intended. By doing so, they can nearly monopolize the CPU.
x??

---

#### Priority Boost Mechanism
Background context: The text introduces a mechanism called "priority boost" where after a certain period $S$, all jobs are moved to the topmost queue to ensure that CPU-bound processes do not starve and interactive processes get proper treatment.

:p How does the priority boost mechanism work?
??x
The priority boost mechanism involves periodically moving all jobs in the system to the highest priority queue. This ensures that even long-running, CPU-bound jobs will receive some CPU time, while interactive jobs are treated correctly after receiving a priority boost.
x??

---

#### Starvation Problem and Priority Boost Solution
Background context: The text highlights the issue of starvation where long-running processes may not get sufficient CPU time if they are competing with short-lived interactive processes. The proposed solution is to periodically move all jobs to the highest priority queue.

:p What problem does the priority boost rule solve?
??x
The priority boost rule addresses the problem of starvation, ensuring that even long-running, CPU-bound jobs will receive some CPU time by moving them to the topmost queue after a certain period $S$. This ensures that interactive processes are also treated properly once they receive a priority boost.
x??

---

#### Example Scenario with Priority Boost
Background context: The text provides an example where a long-running job competes for CPU time with two short-lived, interactive jobs. Without priority boosting, the long-running job gets starved. With priority boosting every 50 ms, the long-running job is guaranteed to make some progress.

:p How does moving all jobs to the highest priority queue after $S$ seconds help in this scenario?
??x
By moving all jobs to the topmost queue after a certain period $S$, the long-running job is guaranteed to receive CPU time periodically. This ensures that it makes some progress, even if only briefly. The interactive jobs will also get their share of CPU time when they are boosted.
x??

---

#### Code Example for Priority Boost
Background context: The text does not provide specific code but suggests a simple rule where all jobs are moved to the topmost queue after every $S$ seconds.

:p How can we implement the priority boost in pseudocode?
??x
Here's a simple pseudocode implementation of the priority boost:
```pseudocode
function prioritizeJobs():
    global S // Time period for prioritization
    while true:
        sleep(S) // Wait for S seconds
        for each job in system:
            moveJobToTopmostQueue(job)
```
This function waits for a fixed time $S$, then moves all jobs to the topmost queue, ensuring that they get some CPU time.
x??

---

#### Security Considerations in Scheduling
Background context: The text emphasizes that scheduling policies can be a security concern, especially in environments where multiple users share resources. Poorly designed or enforced policies can allow one user to harm others.

:p Why is it important for scheduling policies to be secure?
??x
Scheduling policies are crucial for system security because they determine how resources are allocated among competing processes. If not properly designed and enforced, a single user might exploit the scheduler to gain unfair advantages, such as monopolizing CPU time or causing denial-of-service conditions for other users.
x??

---

#### Multi-Level Feedback Queue (MLFQ) Overview
Background context: The text describes an MLFQ scheduling algorithm but does not delve deeply into its implementation. Instead, it focuses on the issues of gaming the scheduler and implementing a priority boost mechanism.

:p What is the main idea behind the multi-level feedback queue (MLFQ)?
??x
The main idea behind the multi-level feedback queue (MLFQ) is to categorize processes based on their behavior over time. Processes start in lower-priority queues and move up or down these queues depending on their CPU usage, I/O activity, etc. This ensures that both long-running jobs and interactive jobs are treated appropriately.
x??

---

#### Summary of Key Concepts
Background context: The text covers several key concepts related to scheduling security, including gaming the scheduler, starvation problems, priority boosts, and multi-level feedback queues.

:p What are the main takeaways from this section?
??x
The main takeaways include understanding that schedulers can be exploited by users ("gaming the scheduler"), recognizing the issue of process starvation in multi-user environments, implementing mechanisms like periodic priority boosts to ensure fair resource distribution, and considering security when designing scheduling policies.
x??

---

#### Voo-doo Constants in Scheduling
Background context explaining the concept. John Ousterhout, a well-regarded systems researcher, referred to certain values in system scheduling as "voo-doo constants" because setting them correctly seemed to require some form of black magic. These values have significant implications for how processes are scheduled and can impact performance.
:p What is the term used by John Ousterhout to describe certain system parameters?
??x
The term used by John Ousterhout to describe these critical but seemingly mysterious parameters is "voo-doo constants." This refers to the idea that setting these values properly requires a level of expertise or intuition that borders on magic.
x??

---
#### Setting Priority Queues in MLFQ
Explanation: The number and size of priority queues, as well as the time slice per queue, are crucial factors in implementing an effective Multi-Level Feedback Queue (MLFQ) scheduler. However, there is no clear formula for setting these parameters; it often requires empirical testing based on workload characteristics.
:p How many priority queues should be used in MLFQ?
??x
The number of priority queues in MLFQ can vary depending on the system and its workloads. There is no one-size-fits-all answer, but typically, a few levels (e.g., 3 or 4) are sufficient to manage different types of processes effectively.
x??

---
#### Time Slice Duration in MLFQ
Explanation: The duration of time slices per queue is another critical parameter that needs careful consideration. Too long a slice can lead to starvation of long-running jobs, while too short a slice can fragment CPU usage and negatively impact interactive tasks.
:p What should be considered when setting the length of a time slice in an MLFQ?
??x
When setting the length of a time slice in an MLFQ, one must balance between avoiding job starvation (by not making slices too long) and ensuring fair sharing of the CPU among all processes (by not making them too short). The ideal duration depends on the nature of the workload.
x??

---
#### Gaming Protection in Scheduling
Explanation: In the original Multi-Level Feedback Queue scheme, Rule 4 allowed jobs to retain their priority by relinquishing the CPU before a time slice expired. This led to potential gaming where processes could manipulate their priorities unfairly. The new rule aims to prevent such manipulation by enforcing strict accounting of CPU usage.
:p How does the new Rule 4 address process gaming in MLFQ?
??x
The new Rule 4 addresses process gaming by requiring that once a job uses up its allocated time slice at any level, regardless of how many times it has relinquished the CPU, its priority is reduced. This prevents processes from manipulating their priorities by issuing I/O just before a time slice ends.
x??

---
#### Tuning MLFQ and Other Issues
Explanation: Tuning an MLFQ scheduler involves making decisions on various parameters such as the number of queues, size of time slices per queue, frequency of priority boosts, etc. These choices significantly impact performance but do not have straightforward answers; they require experience with different workloads.
:p What are some key tuning factors for an MLFQ scheduler?
??x
Key tuning factors for an MLFQ scheduler include the number and size of queues, the duration of time slices per queue, and how frequently to boost priority to prevent starvation. These parameters must be carefully chosen based on the specific workload characteristics.
x??

---
#### Example Code: Priority Queue Management
Explanation: The following pseudocode demonstrates a simple way to manage priority queues in an MLFQ scheduler.
:p Provide an example of pseudocode for managing priority queues in MLFQ.
??x
```java
class Process {
    int currentTimeSlice;
    int priorityLevel;
}

class MLFQScheduler {
    Queue<Process>[] queues; // Array of priority queues
    
    void processTimeSlice(Process p) {
        if (p.currentTimeSlice == 0) { // End of current time slice
            if (p.priorityLevel < queues.length - 1) { // If not at the lowest queue
                p.priorityLevel++; // Promote to next higher priority level
                queues[p.priorityLevel].add(p); // Add process to new queue
            } else {
                // Handle processes that have exhausted all time slices
                handleStarvation(p);
            }
        } else {
            p.currentTimeSlice--; // Decrease remaining time slice
            // Process execution logic here
        }
    }

    void handleStarvation(Process p) {
        // Implement starvation handling logic, e.g., move to a lower priority level
    }
}
```
x??

---

#### Ousterhout’s Law
Background context explaining the concept. Most MLFQ (Multilevel Feedback Queue) variants allow for varying time-slice lengths across different queues. High-priority queues are typically given short time slices, while low-priority queues have longer time slices to handle long-running CPU-bound jobs.
:p What is Ousterhout’s Law in the context of scheduling?
??x
Ousterhout's Law states that high-priority queues should have shorter time slices because they deal with interactive jobs that require quick response times. Conversely, low-priority queues can use longer time slices to accommodate long-running CPU-bound tasks.
For example:
- High-priority queue: 10 ms or fewer (e.g., for interactive jobs)
- Low-priority queue: 100 ms or more (e.g., for CPU-bound jobs)
??x

---

#### Solaris MLFQ Implementation
Background context explaining the concept. The Solaris MLFQ implementation, specifically the Time-Sharing scheduling class (TS), is configurable through a set of tables that determine time-slice lengths and priority boosts.
:p How is the Solaris MLFQ implemented?
??x
The Solaris TS scheduler uses configuration tables to define how priorities change over a process's lifetime. By default, it has 60 queues with increasing time-slice lengths from 20 ms (highest priority) to several hundred milliseconds (lowest). The priority of jobs is boosted roughly every second.
For example:
```java
public class TSConfig {
    static int[][] timeSliceTable = new int[60][];
    // Initialize the table with appropriate values
}
```
??x

---

#### Other MLFQ Schedulers
Background context explaining the concept. Different MLFQ schedulers use various methods to determine priority levels and time slices, such as mathematical formulas or decayed usage models.
:p How do other MLFQ schedulers differ from Solaris’s TS scheduler?
??x
Other MLFQ schedulers may not use a table like Solaris's TS but instead employ mathematical formulas to calculate the current priority level of a job based on its CPU usage. For example, FreeBSD version 4.3 uses a formula that decays usage over time to determine the priority.
For instance:
```java
public class OtherMLFQ {
    static int getPriorityLevel(double cpuUsage) {
        // Formula to calculate priority based on CPU usage and decayed value
        return (int)((cpuUsage - DECAY_CONSTANT) * SCALING_FACTOR);
    }
}
```
??x

---

#### Priority Reservations and User Advice
Background context explaining the concept. Some schedulers reserve the highest priority levels for system use, preventing user jobs from accessing these levels. Additionally, users can provide advice to influence scheduling decisions.
:p What are some features related to priority reservations and user advice in MLFQ?
??x
Some schedulers reserve the highest priority levels exclusively for operating system work, ensuring that typical user processes cannot reach these high-priority levels. Users can set job priorities using tools like `nice`, which adjusts the process's niceness value.
For example:
```bash
# Set a process's nice value
nice -n 10 ./myprocess
```
??x

---

#### Using Advice in Scheduling
Background context explaining the concept. The operating system often provides interfaces for users to give hints or advice about scheduling decisions, which can improve overall performance.
:p How does using user advice help in MLFQ?
??x
Using user advice allows administrators or end-users to provide hints that the OS can use when making scheduling decisions. For instance, `nice` is a command-line utility that enables users to adjust process priorities, thereby influencing the likelihood of a job running at any given time.
For example:
```bash
# Adjusting priority with nice
nice -n 10 ./myprocess
```
??x

#### Multi-Level Feedback Queue (MLFQ) Overview
Background context explaining the concept. The MLFQ is a scheduling approach that uses multiple levels of queues to manage different types of jobs based on their priorities and execution behavior over time. Priority adjustments are made dynamically using feedback mechanisms, such as observing how a job behaves in the system.

:p What is the Multi-Level Feedback Queue (MLFQ)?
??x
The MLFQ is a scheduling algorithm that employs multiple levels of queues to prioritize tasks based on their performance and behavior within the system over time. It adjusts priorities through feedback loops to optimize overall system performance for both short- and long-running jobs.
x??

---

#### Rule 1: Higher Priority Runs
This rule dictates that if Job A has a higher priority than Job B (Priority(A) > Priority(B)), Job A will run while Job B does not.

:p What is the first scheduling rule in MLFQ?
??x
If the priority of Job A is greater than the priority of Job B, then Job A runs and Job B does not. This ensures that higher-priority jobs are executed before lower-priority ones.
x??

---

#### Rule 2: Round-Robin for Equal Priorities
This rule states that if two jobs have the same priority (Priority(A) = Priority(B)), they should run in a round-robin fashion using the time slice or quantum length of their current queue.

:p What happens when two jobs have the same priority?
??x
When two jobs have the same priority, they will be scheduled to run in a round-robin manner according to the time slice (quantum length) defined for that particular queue. This ensures fairness between equally prioritized tasks.
x??

---

#### Rule 3: Initial Job Placement
This rule places newly arrived jobs at the highest priority level (topmost queue).

:p How are new jobs placed in MLFQ?
??x
Newly arriving jobs are initially placed at the highest priority level, which is usually the topmost queue. This placement ensures that new tasks receive immediate attention.
x??

---

#### Rule 4: Priority Reduction and Queue Movement
This rule reduces a job's priority (moves it down one queue) once its time allotment in the current queue expires, regardless of how many times it has yielded control.

:p What happens when a job uses up its time slice?
??x
When a job exhausts its allocated time slice at a given priority level, its priority is reduced by moving it to the next lower queue. This adjustment ensures that long-running jobs eventually give way to higher-priority tasks.
x??

---

#### Rule 5: Periodic Queue Reordering
This rule suggests periodically reordering all jobs in the system back to the highest priority (topmost) queue after a certain time period S.

:p What does the fifth MLFQ rule state?
??x
After a specified time period S, all jobs in the system are moved back to the topmost queue. This ensures that old and new tasks have an equal chance of being executed.
x??

---

#### Benefits of MLFQ
MLFQ achieves excellent overall performance for short-running interactive jobs while maintaining fairness and progress for long-running CPU-intensive workloads by dynamically adjusting priorities based on job behavior.

:p What are the benefits of using MLFQ?
??x
The primary benefits of MLFQ include delivering outstanding performance for short-running, interactive tasks similar to Shortest Job First (SJF) or Shortest Time-to-Completion First (STCF). Additionally, it ensures fairness and progress for long-running CPU-intensive workloads. Its adaptive nature allows it to balance between these extremes effectively.
x??

---

#### Real-World Usage of MLFQ
Many operating systems, including BSD UNIX derivatives, Solaris, and Windows NT and subsequent versions, use a form of the MLFQ as their base scheduler.

:p Which operating systems commonly use MLFQ?
??x
Operating systems such as BSD UNIX derivatives, Solaris, and various versions of Windows (including NT) utilize a variant of the Multi-Level Feedback Queue (MLFQ) for scheduling. This widespread adoption highlights its effectiveness in managing diverse workloads.
x??

---

#### References for Further Reading
Additional resources include academic papers and books that provide detailed insights into MLFQ and related scheduling algorithms.

:p Where can I find more information on MLFQ?
??x
For further reading, consider the following references:
- "Multilevel Feedback Queue Scheduling in Solaris" by Andrea Arpaci-Dusseau.
- "The Design of the UNIX Operating System" by M.J. Bach.
- "An Experimental Time-Sharing System" by F.J. Corbato et al.
- "Inside Windows NT" by Helen Custer and David A. Solomon.
- "An Analysis of Decay-Usage Scheduling in Multiprocessors" by D.H.J. Epema.

These resources offer comprehensive information on the implementation, history, and performance analysis of MLFQ.
x??

---
#### 4.3BSD UNIX Operating System Book
Background context: This is a classic book about the design and implementation of the 4.3BSD Unix operating system, written by four influential figures behind BSD (S.J. Leffler, M.K. McKusick, M.J. Karels, J.S. Quarterman). While later versions are more up to date, this particular version is considered beautiful.
:p What book about 4.3BSD UNIX Operating System was written by the key people behind BSD?
??x
The Design and Implementation of the 4.3BSD UNIX Operating System by S.J. Leffler, M.K. McKusick, M.J. Karels, J.S. Quarterman.
x??

---
#### Solaris Internals Book
Background context: This book provides in-depth knowledge about the Solaris operating system and its architecture (Solaris 10 and OpenSolaris). It covers how it works, making it a valuable resource for understanding Solaris internals.
:p What is the title of the book that describes Solaris 10 and OpenSolaris kernel architecture?
??x
Solaris Internals: Solaris 10 and OpenSolaris Kernel Architecture by Richard McDougall.
x??

---
#### John Ousterhout’s Home Page
Background context: This is a personal home page by renowned Professor Ousterhout. The co-authors of another book got to know each other through their graduate school classes with him, eventually leading to marriage and having children. The content relates closely to operating systems.
:p Which famous professor’s home page connects the co-authors of another book?
??x
John Ousterhout's Home Page at www.stanford.edu/˜ouster/.
x??

---
#### Informed Prefetching and Caching Paper
Background context: This paper discusses innovative ideas in file systems, including how applications can provide hints to the operating system about their intended access patterns. It’s an interesting read for those interested in file system design.
:p What is the title of the paper on informed prefetching and caching?
??x
Informed Prefetching and Caching by R.H. Patterson, G.A. Gibson, E. Gins ting, D. Stodolsky, J. Zelenka.
x??

---
#### Scheduling Analysis for Distributed Storage Systems
Background context: This recent work examines the challenges of scheduling I/O requests in modern distributed storage systems like Hive/HDFS, Cassandra, MongoDB, and Riak. It highlights the difficulty of managing such systems without careful design.
:p What is the title of the paper that analyzes schedulability in distributed storage systems?
??x
Principled Schedulability Analysis for Distributed Storage Systems using Thread Architecture Models by Suli Yang, Jing Liu, Andrea C. Arpaci-Dusseau, Remzi H. Arpaci-Dusseau.
x??

---
#### MLFQ Scheduler Simulation Program
Background context: This program (`mlfq.py`) allows you to experiment with the Multi-Level Feedback Queue (MLFQ) scheduler presented in a chapter of another book. The objective is to understand how it behaves under different conditions.

:p What Python program simulates the MLFQ scheduler, allowing for experimentation?
??x
`mlfq.py`
x??

---
#### Homework Questions on MLFQ Scheduler

1. **Run Randomly-Generated Problems with Two Jobs and Two Queues**
   Background context: This involves running a few randomly-generated problems involving two jobs and two queues to understand how the scheduler traces execution.
   
   :p What is the first question in the homework about the MLFQ scheduler?
   ??x
   Run a few randomly-generated problems with just two jobs and two queues; compute the MLFQ execution trace for each. Make your life easier by limiting the length of each job and turning off I/Os.
   x??

---
#### Configuring Scheduler Parameters

2. **Configure Scheduler to Behave Like Round-Robin**
   Background context: This question requires understanding how to adjust scheduler parameters so that it behaves like a round-robin scheduler.

   :p How would you configure the scheduler parameters to behave just like a round-robin scheduler?
   ??x
   To configure the scheduler parameters to behave like a round-robin scheduler, you would need to set up appropriate time slices and priorities in each queue such that no job gets preempted until it has completed its quantum.
   x??

---
#### Crafting Workload for Gaming Scheduler

3. **Craft a Workload with Two Jobs**
   Background context: This involves creating a specific workload where one job can dominate the CPU by exploiting certain rules.

   :p Craft a workload with two jobs and scheduler parameters so that one job takes advantage of older rules to game the scheduler.
   ??x
   Craft a workload with two jobs such that one job exploits Rules 4a and 4b (turned on with the -S flag) to obtain 99 percent of the CPU over a particular time interval. The other job should be designed to allow this behavior.
   x??

---
#### Boosting Jobs Back to Highest Priority

4. **Boost Jobs in High Queue**
   Background context: This question explores how often jobs need to be boosted back to the highest priority level to ensure that a single long-running job gets CPU time.

   :p Given a system with a quantum length of 10 ms in its highest queue, how frequently would you have to boost jobs back to the highest priority level to guarantee that a single long-running job gets at least 5 percent of the CPU?
   ??x
   To ensure that a single long-running job gets at least 5 percent of the CPU, you would need to calculate how many time intervals (each 10 ms) are required for the long-running job to get its share. Since there are 200 such intervals in one second, boosting back every 20 intervals should guarantee that the job gets approximately 5 percent of the CPU.
   x??

---
#### Effect of -I Flag

5. **Effect of -I Flag**
   Background context: This flag changes how jobs are added to a queue after completing I/O.

   :p How does the `-I` flag affect the scheduling simulator?
   ??x
   The `-I` flag in the scheduling simulator changes where a job is added at the end of a queue once it has completed its I/O. Playing around with different workloads can help observe the effect of this flag on the scheduler's behavior.
   x??

---

