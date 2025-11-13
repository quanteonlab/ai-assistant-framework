# Flashcards: Parallel-and-High-Performance-Computing_processed (Part 51)

**Starting Chapter:** 15.3 Submitting your first batch script

---

#### Batch Schedulers: Overview
Background context explaining the importance of batch schedulers in managing Beowulf clusters. In the late 1990s, Beowulf clusters emerged as a way to build computing clusters using commodity hardware. However, without software control and management, these clusters would not be productive resources.
:p What is the significance of batch schedulers in Beowulf clusters?
??x
Batch schedulers are crucial because they manage the workload on busy clusters efficiently. They ensure that jobs are allocated to nodes based on policies defined by queue rules, thus maximizing resource utilization and job throughput. Without a batch system, multiple users can lead to inefficiencies and conflicts.
```python
# Example of a simple scheduling rule in pseudocode
def schedule_job(job):
    if job.memory <= max_memory and job.time <= max_time:
        allocate_node_to_job(job)
    else:
        add_job_to_queue(job)

# Example usage
schedule_job(job1)  # Checks if the job can be scheduled directly or added to a queue
```
x??

---

#### Front-End Nodes in Batch Systems
Background context explaining that front-end nodes, also called login nodes, are where users typically log into the cluster. These nodes manage interactions with back-end nodes and have different policies compared to back-end nodes.
:p What is the role of front-end nodes in batch systems?
??x
Front-end nodes serve as entry points for users to access the cluster. They handle user commands, job submissions, and monitor system load. Users should be aware that these nodes can get busy during peak times, which might affect their job submission times or execution speeds.
```java
// Example of monitoring front-end node load in pseudocode
public class FrontEndNodeMonitor {
    public boolean isLightlyLoaded() {
        // Check if the number of running jobs on this node is below a threshold
        return getNumberOfRunningJobs() < 10;
    }
}
```
x??

---

#### Managing Back-End Nodes with Queues
Background context explaining that back-end nodes are organized into queues, each with specific policies for job size and runtime. This setup helps in efficient resource allocation.
:p How do back-end nodes manage jobs?
??x
Back-end nodes manage jobs by allocating them based on predefined queue rules. These rules typically consider the number of processors or memory needed and the maximum runtime allowed. By organizing nodes into queues, the system ensures fair usage and maximizes overall efficiency.
```python
# Example of a queue in pseudocode
class JobQueue:
    def __init__(self, max_processors, max_time):
        self.max_processors = max_processors
        self.max_time = max_time

    def can_accept_job(self, job):
        return job.processors <= self.max_processors and job.time <= self.max_time
```
x??

---

#### Courteous Use of Busy Clusters
Background context explaining the importance of being considerate when working on busy clusters. Common practices include monitoring load, avoiding heavy file transfers, and using appropriate nodes for different tasks.
:p What are common courteous practices when working on a busy cluster?
??x
Common courteous practices include:
- Monitoring front-end node load to avoid overloading it (use `top` command).
- Avoiding running large file transfer jobs on the front end, as they can impact other users’ jobs.
- Compiling code on the appropriate nodes; some sites prefer compilation on back-end nodes for performance reasons.

By following these practices, you ensure that your usage of the cluster does not interfere with others and helps maintain a productive environment.
```java
// Example of moving to a lightly loaded front end in pseudocode
public class NodeSelector {
    public String selectLightlyLoadedNode() {
        // Check each node's load and return one that is under a certain threshold
        for (String nodeName : nodes) {
            if (isNodeLightlyLoaded(nodeName)) {
                return nodeName;
            }
        }
        return null;  // No available lightly loaded nodes found
    }

    private boolean isNodeLightlyLoaded(String nodeName) {
        // Logic to determine if the node has a low load
        return true;  // Placeholder logic
    }
}
```
x??

---

#### Node Usage Management
Background context explaining how node usage can impact cluster stability and job scheduling. Nodes should not be tied up with batch interactive sessions for extended periods, especially when attending meetings.

:p How can one ensure efficient use of nodes during non-meeting times?
??x
To ensure efficient use of nodes during non-meeting times, users should avoid tying up nodes with batch interactive sessions. Instead, they can export an X terminal or shell from their initial session to maintain connectivity without monopolizing a node. This approach helps in keeping the cluster responsive for other tasks.
```shell
# Example: Exporting an X11 terminal from an SSH session
ssh -X user@node
```
x??

---

#### Queue Selection Based on Workload
Explaining how to choose appropriate queues based on job requirements, especially for light work and debugging.

:p What should be considered when selecting a queue for running small tasks?
??x
For light work or shared usage that allows over-subscription, users should look for dedicated queues. These queues are typically configured to handle less resource-intensive tasks without the need for heavy reservations, thus allowing more flexible and efficient use of cluster resources.
```shell
# Example: Checking available queues
qstat -Q
```
x??

---

#### Big Job Management
Discussing strategies for managing large jobs in a batch system, including queue management and non-work hours scheduling.

:p What is the recommended approach for running big parallel jobs?
??x
For big parallel jobs, it is recommended to use the back-end nodes through the batch system queues. Additionally, keep the number of jobs in the queue small to avoid monopolizing resources. Running large jobs during non-work hours can also help other users get interactive nodes for their work.
```shell
# Example: Submitting a job with a specific time limit and dependency on previous job
qsub -l walltime=24:00:00 -W depend=afterok:prev_job_id script.sh
```
x??

---

#### Storage Management
Discussing best practices for managing storage, including file system usage, periodic purging, and regular cleanup.

:p What are the key points in managing large files on a cluster?
??x
When managing large files, it is crucial to store them in appropriate directories like parallel filesystems, scratch, project, or work directories. Files should be moved to long-term storage regularly, and users must be aware of purging policies that may clear scratch directories periodically. Keeping file systems below 90% full ensures optimal performance.
```shell
# Example: Checking disk usage
df -h
```
x??

---

#### Cluster Stability and Fair-Share Scheduling
Explaining the importance of avoiding heavy front-end node usage to prevent system instabilities, and how resource allocations may be prioritized using fair-share scheduling.

:p How can users avoid causing instability in the cluster?
??x
To avoid causing instability in the cluster, users should minimize their use of the front-end nodes. Overusing these nodes can lead to scheduling issues for back-end nodes, impacting overall system stability. Additionally, resource allocations often include fair-share policies that prioritize jobs based on usage patterns.
```shell
# Example: Checking current job status and node utilization
top; qstat
```
x??

---

#### Following Cluster Policies Ethically
Discussing the importance of adhering to cluster policies and avoiding gaming the system for personal gain.

:p What ethical considerations should users keep in mind when using a cluster?
??x
Users must follow both the letter and spirit of cluster policies. Gaming the system by abusing resources or queue rules is not advisable as it affects fellow users who are also trying to complete their work. Optimizing code and file storage can help maximize efficiency without compromising others' access.
```shell
# Example: Submitting a well-formatted job script
#!/bin/bash
#SBATCH --job-name=example_job
#SBATCH --mail-type=ALL
#SBATCH --output=output.log

module load necessary_modules

# Your commands here
```
x??

---

#### Batch Job Submission Strategies
Discussing strategies for submitting multiple jobs in an efficient manner, including the use of shell scripts and job dependencies.

:p How can users submit multiple jobs efficiently without monopolizing queue resources?
??x
Users should submit a limited number of jobs initially (e.g., 10) and resubmit more as needed. This approach helps maintain balance within the queue system. Shell scripts or batch dependency techniques can automate this process, ensuring jobs are submitted in an organized manner.
```shell
# Example: Simple shell script to submit multiple jobs
#!/bin/bash

for i in {1..10}
do
    qsub -t $i my_job_script.sh
done
```
x??

---

#### Checkpointing for Long-Running Jobs
Explaining the importance of checkpointing for long-running batch jobs to optimize resource usage.

:p What is the purpose of implementing checkpointing for long jobs?
??x
Checkpointing is crucial for managing long-running batch jobs by catching termination signals or using wall clock timings. It allows subsequent jobs to resume from a saved state, making optimal use of available time and resources. This technique ensures that jobs can complete even if interrupted.
```shell
# Example: Implementing checkpointing in a script
#!/bin/bash

for ((i=1; i<=20; i++))
do
    echo "Running iteration $i"
    # Simulate some work
    sleep 5s
    
    # Checkpointing logic (example pseudo-code)
    if [ $((i % 5)) -eq 0 ]; then
        echo "Checkpointing at iteration $i"
        # Save state to disk or file system
    fi
done
```
x??

---

#### Batch Scripting Basics
Batch systems require a structured approach to job submission, different from ad-hoc job launches. Understanding these systems leads to more efficient resource utilization.

:p What is the purpose of using batch scripts in computing environments?
??x
The primary purpose of using batch scripts is to manage and schedule jobs efficiently, ensuring better use of computational resources. Batch scripts allow for automated job execution, especially useful for long-running or overnight tasks that require no user interaction.
x??

---

#### Salloc Command Example
The `salloc` command allocates nodes and logs into them, initiating a job session on the compute cluster.

:p How do you initiate a salloc request for two nodes with 32 processors each, running for one hour?
??x
To initiate an `salloc` request for two nodes with 32 processors each and run a job for one hour, use the following command:

```
frontend> salloc -N 2 -n 32 -t 1:00:00
```

This command allocates two compute nodes (`-N 2`) and requests 32 processors in total (`-n 32`), with a job duration of one hour (`-t 1:00:00`). Note the case sensitivity between `N` (number of nodes) and `n` (number of processes).
x??

---

#### Running Applications on Allocated Nodes
Once nodes are allocated, you can run applications as required.

:p How do you launch a parallel application using mpirun after allocating nodes with salloc?
??x
After allocating nodes with the `salloc` command, you can launch a parallel application like this:

```
computenode22> mpirun -n 32 ./my_parallel_app
```

This command uses `mpirun` to start the parallel application `./my_parallel_app` on 32 processes. Note that the specific commands and options might vary depending on your system.
x??

---

#### Transitioning to Batch Scripts
Batch scripts are useful for running jobs without user interaction, allowing automation and resubmission in case of failures.

:p How do you convert an interactive job into a batch script?
??x
To convert an interactive job into a batch script, first create a text file with the necessary `sbatch` directives. Here’s how to modify the previous example:

```bash
#SBATCH -N 2       # Request two nodes
#SBATCH -n 32      # Request 32 processors
#SBATCH -t 1:00:00 # Run for one hour

mpirun -n 32 ./my_parallel_app
```

Save this as `my_batch_job` and submit it using:

```
frontend> sbatch my_batch_job
```

This script specifies the number of nodes, processors, and runtime, followed by the command to run the application.
x??

---

#### System Modes in Batch Systems
Batch systems operate in two main modes: interactive mode for development and testing, and batch usage mode for long-running jobs.

:p What are the two basic system modes used in batch systems?
??x
Batch systems typically use two basic modes:
1. **Interactive Mode**: Used for program development, testing, or short jobs.
2. **Batch Usage Mode**: Commonly used for submitting longer production jobs that run unattended.

The interactive mode is often used on the front end of the cluster where users log in and interact with nodes directly. Batch usage mode involves submitting job scripts to be executed automatically by the batch scheduler.
x??

---

#### Batch Scheduler Overview
Slurm is a popular batch scheduler used for managing parallel computing jobs. It helps organize and manage compute resources efficiently by scheduling jobs to run at specific times and on designated nodes. The options provided allow for precise control over job execution, resource allocation, and output redirection.

:p What are the key features of Slurm as described in the text?
??x
Slurm provides various options such as specifying node count, number of tasks (processors), wall time, job names, and error/output file locations. It also supports exclusive use or oversubscription of resources to optimize performance based on specific requirements.

```bash
# Example SBATCH directives for a Slurm script
#SBATCH -N 1      # Requests one compute node
#SBATCH -n 4      # Requests four processors
#SBATCH -t 01:00:00 # Requests the job to run for up to 1 hour
```
x??

---

#### SBATCH Directive Examples
SBATCH directives are used at the beginning of a Slurm batch script to specify job requirements. These can include resource allocation, execution time limits, and output files.

:p What does the `#SBATCH -N <number>` directive do in a Slurm batch script?
??x
The `#SBATCH -N <number>` directive specifies the number of nodes required for the job. For example, `-N 2` would request two compute nodes. This is crucial as it determines how many physical resources are allocated to your parallel application.

```bash
#SBATCH -N 1      # Requests one node
```
x??

---

#### SBATCH Directive for Task Count
The `#SBATCH -n <count>` directive allows you to specify the number of tasks or processors that will be used by the job. This is particularly important in MPI applications where you need to define how many processes should run.

:p What does the `#SBATCH -n` directive control?
??x
The `#SBATCH -n <count>` directive controls the number of tasks (processors) allocated for a job. In an MPI context, it determines how many instances of the parallel application will be launched across available cores or nodes.

```bash
#SBATCH -n 4      # Requests four processors
```
x??

---

#### Time Limit Specification in Slurm
The `#SBATCH -t <time>` directive is used to set a time limit for the job. This ensures that the job runs for no longer than the specified duration, preventing it from running indefinitely.

:p How do you specify the maximum runtime of a job using SBATCH?
??x
You can specify the maximum runtime of a job using the `#SBATCH -t <time>` directive. The time is given in `hr:min:sec` format. For example, to request an hour-long job, you would use `-t 01:00:00`.

```bash
#SBATCH -t 01:00:00 # Requests a one-hour runtime
```
x??

---

#### Job Naming and Output Files
Slurm allows you to name your jobs and direct their output (both standard out and error) to specific files. This is useful for tracking job progress and debugging.

:p How do you specify the names of your job and where it should write its outputs?
??x
You can name your job using `#SBATCH --job-name=<name>` and redirect both standard out and error output to specified files with `--output` and `--error`. For example:

```bash
#SBATCH --job-name=my_job      # Names the job "my_job"
#SBATCH --output=run.out       # Redirects standard output to run.out
#SBATCH --error=run.err        # Redirects standard error to run.err
```
x??

---

#### OverSubscription and Exclusive Use
OverSubscription allows more tasks than available nodes, while exclusive use ensures that only one job can be run per node. These options are useful for optimizing resource utilization.

:p What do the `--exclusive` and `--oversubscribe` options in Slurm control?
??x
The `--exclusive` option requests exclusive use of a node or set of resources, meaning no other jobs will run on these nodes during your job's execution. The `--oversubscribe` option allows more tasks than the number of processors available, which can be useful for filling idle time.

```bash
#SBATCH --exclusive  # Requests exclusive resource use
#SBATCH --oversubscribe  # Allows oversubscription of resources
```
x??

---

#### Example Batch Script Submission
This example demonstrates how to submit a Slurm batch script and its components. It includes SBATCH directives, the command to be executed, and the submission process.

:p How do you submit a job using a Slurm batch script?
??x
To submit a job using a Slurm batch script, save the script with appropriate SBATCH directives and commands, then use `sbatch` followed by the filename. For example:

```bash
sbatch my_batch_job.slurm  # Submits the specified batch script
```
You can also run this interactively if necessary, using `salloc` to allocate resources directly.

```bash
frontend> salloc -N 1 -n 4 -t 01:00:00
computenode22> mpirun -n 4 ./testapp  # Runs the application on allocated nodes
```
x??

---

---
#### Slurm Node Request Command
Background context: Slurm provides commands to request nodes with specific characteristics. One such command is `--mem=<#>` which helps in requesting large memory nodes based on the specified size in MB.

:p How do you use the `--mem` option in Slurm to request a node with at least 32GB of memory?
??x
To request a node with at least 32GB of memory, you would use the following command:

```bash
srun --mem=32000 ./your_command
```

Here, `--mem=32000` requests a node that has at least 32GB (32000MB) of available memory. This ensures that your job runs on a machine with sufficient memory resources.
x??

---
#### PBS Batch Script Example
Background context: The provided text explains the structure and usage of a PBS batch script, which is used for submitting jobs to the PBS batch scheduler.

:p What is the equivalent PBS batch script for requesting 4 processes on one node with a maximum wall time of 1 hour?
??x
Here’s an example PBS batch script that requests 4 processes on one node and runs your job within a maximum wall time of 1 hour:

```bash
#!/bin/sh
#PBS -l nodes=1:ppn=4
#PBS -l walltime=01:00:00

mpirun -n 4 ./testapp &> run.out
```

In this script:
- `#PBS -l nodes=1:ppn=4` specifies that the job requires one node with 4 processes per node.
- `#PBS -l walltime=01:00:00` sets the maximum allowable runtime for the job to 1 hour.

You can submit this script using the following command:
```bash
qsub first_pbs_batch_job
```
x??

---
#### Slurm Interactive Allocation Command
Background context: The `salloc` and `sbatch` commands in Slurm are used to allocate nodes for batch jobs. `salloc` is particularly useful for interactive allocations where you need immediate access to a node.

:p How do you request 1 node with 32 tasks using the `salloc` command?
??x
To request one node with 32 tasks using the `salloc` command, you would use:

```bash
salloc --nodes=1 -n 32
```

This command allocates a single node and reserves it for your interactive session. The `-n 32` option indicates that you need 32 cores or processes on this node.
x??

---
#### PBS Job Submission with `qsub`
Background context: The `qsub` command in PBS is used to submit jobs to the batch scheduler, allowing them to run asynchronously.

:p How do you submit a job using the PBS `qsub` command?
??x
To submit a job using the PBS `qsub` command, follow this example:

```bash
qsub first_pbs_batch_job
```

Here, `first_pbs_batch_job` is your PBS batch script file. This command submits the specified script to the PBS scheduler for execution.
x??

---
#### Slurm Job Status Check with `squeue`
Background context: The `squeue` command in Slurm provides information about jobs currently queued or running on the system.

:p How do you use the `squeue` command to check the status of your jobs?
??x
To check the status of your jobs using the `squeue` command, simply run:

```bash
squeue
```

This command displays a list of all jobs in the queue along with their statuses. You can filter or specify additional options if needed.
x??

---
#### PBS Job Output Redirection with `-o` and `-e`
Background context: The `-o` and `-e` options in PBS are used to redirect standard output (stdout) and standard error (stderr) to specified files, respectively.

:p How do you redirect both stdout and stderr to a single file using the `qsub` command?
??x
To redirect both stdout and stderr to a single file, you can use the `-o` option with the `qsub` command as follows:

```bash
qsub -o run.out first_pbs_batch_job
```

Here, `run.out` is the filename where the combined output of your job will be stored. This ensures that both stdout and stderr are captured in a single file for easier monitoring.
x??

---

---
#### squeue Command Usage
The `squeue` command is used to view information about running and pending jobs on a Slurm cluster. It provides details such as job ID, partition, name, user, state, time, nodes, and node list.
:p What does the `squeue` command display?
??x
`squeue` displays detailed information about running and pending jobs including:
- JOBID: The unique identifier for each job.
- PARTITION: The partition (or queue) on which the job is submitted.
- NAME: A name assigned to the job by the user.
- USER: The user who submitted the job.
- ST: State of the job, such as PD (Pending), R (Running).
- TIME: Time the job has been in its current state or running time.
- NODES: Number of nodes allocated for the job.
- NODELIST: List of nodes where the job is running.

Example usage:
```bash
frontend> squeue
```
x??

---
#### srun Command Usage
The `srun` command is used to run a parallel application, similar to `mpiexec`. It can be used as a replacement for MPI applications and provides additional options like resource affinity.
:p What does the `srun` command allow you to do?
??x
`srun` allows running parallel applications with enhanced capabilities such as specifying the number of nodes, tasks, and CPU binding. It offers flexibility in allocating resources based on the needs of the application.

Example usage:
```bash
frontend> srun -N 1 -n 16 --cpu-bind=cores my_exec
```
This command runs `my_exec` using 16 tasks on a single node with cores bound as specified.
x??

---
#### scontrol Command Usage
The `scontrol` command is used to view or modify Slurm components. It can show detailed information about jobs, partitions, nodes, etc., and make changes to job submissions, reservations, and more.
:p What does the `scontrol` command allow you to do?
??x
`scontrol` allows viewing or modifying various aspects of a Slurm cluster, including:
- Viewing job details: `scontrol show job <SLURM_JOB_ID>`
- Modifying job requirements: Setting node exclusions, time limits, etc.

Example usage for showing job details:
```bash
frontend> scontrol show job 35456
```
This command provides detailed information about the specified job ID.
x??

---
#### qsub Command Usage
The `qsub` command submits a batch job to Slurm. It can be configured with options like interactivity, waiting for completion, and more. The equivalent in PBS is using directives within the script.
:p What does the `qsub` command do?
??x
`qsub` is used to submit a batch job to the Slurm scheduler. Options such as interactivity (`-I`), blocking until completion (`-W block=true`), and more can be specified in the job submission script.

Example usage for an interactive session:
```bash
frontend> qsub -I
```
This command starts an interactive shell session on a compute node.
x??

---
#### qdel Command Usage
The `qdel` command is used to delete a batch job that has been submitted. It can be used with the job ID as an argument.
:p How do you use the `qdel` command?
??x
To delete a batch job, use the `qdel` command followed by the job ID.

Example usage:
```bash
frontend> qdel <job ID>
```
This command will remove the specified job from the queue.
x??

---
#### qsig Command Usage
The `qsig` command sends a signal to a running or pending batch job. It can be used for various purposes, such as debugging or managing jobs.
:p What does the `qsig` command do?
??x
The `qsig` command allows sending signals to running or pending batch jobs in Slurm.

Example usage:
```bash
frontend> qsig 23 56
```
This command sends a signal (numbered as per system conventions, e.g., 1 for termination) to the specified job IDs.
x??

---
#### qstat Command Usage
The `qstat` command is used to view the status of running or pending batch jobs. It provides information about the job's state and resource usage.
:p What does the `qstat` command display?
??x
`qstat` displays the status of running or pending batch jobs, including:
- Job ID: Unique identifier for each job.
- User: The user who submitted the job.
- Queue: Partition or queue where the job is submitted.
- Jobname: Name assigned to the job by the user.
- Sess: Session information.
- NDS: Number of directives.
- TSK: Number of tasks.
- Mem: Memory requested for the job.
- Time: Time required for the job.

Example usage:
```bash
frontend> qstat -u jrr
```
This command shows the status of jobs submitted by user `jrr`.
x??

---
#### qmsg Command Usage
The `qmsg` command sends a message to a running batch job. It can be used for debugging or providing information to the application during execution.
:p How do you use the `qmsg` command?
??x
To send a message to a running batch job, use the `qmsg` command followed by the message and job ID.

Example usage:
```bash
frontend> qmsg "message to standard error" 56
```
This command sends a message ("message to standard error") to job with ID `56`.
x??

---
#### Slurm Environment Variables in Batch Scripts
Slurm provides several environment variables that can be useful in batch scripts for resource allocation and monitoring. These include `SLURM_NTASKS`, `SLURM_CPUS_ON_NODE`, `SLURM_JOB_CPUS_PER_NODE`, etc.
:p What are some important Slurm environment variables for batch scripts?
??x
Important Slurm environment variables for batch scripts include:
- `SLURM_NTASKS`: Total number of tasks or processors requested (formerly known as `SLURM_NPROCS`).
- `SLURM_CPUS_ON_NODE`: Number of CPUs on the allocated node.
- `SLURM_JOB_CPUS_PER_NODE`: Number of CPUs requested for each node.
- `SLURM_JOB_ID`: ID of the current job.
- `SLURM_JOB_NODELIST`: List of nodes allocated to the job.
- `SLURM_JOB_NUM_NODES`: Total number of nodes in the job.
- `SLURM_SUBMIT_DIR`: Directory from which the job was submitted.
- `SLURM_TASKS_PER_NODE`: Number of tasks per node.

These variables can be used within batch scripts to dynamically adjust resource allocation based on the environment.
x??

---

#### Batch Restart Script Concept
Background context: The provided script is designed to automatically restart long-running jobs that are subject to time limits imposed by batch schedulers like Slurm. This approach involves periodically saving the state of the job (checkpointing) and resubmitting a new job that can pick up from where the previous one left off.
:p What does this script do?
??x
This script handles automatic restarts for long-running jobs by periodically checkpointing their state and re-submitting them if they exceed the allocated time. It ensures that even if a job is terminated prematurely, it can be resumed from the last known state.
```sh
# Example usage of batch_restart.sh
sbatch <batch_restart.sh
```
x??

---

#### Signal Handling in Application Code
Background context: The application code demonstrates how to handle signals sent by the batch system and how to perform checkpointing. Signals are used to notify the application that its time is nearly up, allowing it to save its state before termination.
:p How does the application catch the signal?
??x
The application catches the signal using a signal handler function `batch_timeout`. This function sets a global variable `batch_terminate_signal` when it receives the signal. The main loop of the application checks this variable periodically and exits if set, allowing for graceful shutdown.
```c
static int batch_terminate_signal = 0;

void batch_timeout(int signum){
    printf("Batch Timeout : %d",signum);
    batch_terminate_signal = 1;
    return;
}
```
x??

---

#### Iteration Control in Application Code
Background context: The application uses iteration control to simulate computational work and checkpointing. It writes out checkpoints at regular intervals and handles restarts by reading the last known state.
:p How does the application manage iterations?
??x
The application manages iterations through a loop that simulates computational work using `sleep(1)`. Checkpoints are written every 60 iterations, and the current iteration number is stored in a file named `RESTART` upon termination. The application can be restarted by reading this iteration number.
```c
for (int it=itstart; it < 10000; it++){
    sleep(1);
    if ( it % 60 == 0 ) {
        // Write out checkpoint file
    }
    int terminate_sig = batch_terminate_signal;
    MPI_Bcast(&terminate_sig, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if ( terminate_sig ) {
        // Write out RESTART and special checkpoint file
    }
}
```
x??

---

#### Checkpointing and Restart File Handling
Background context: The application writes out a checkpoint file every 60 iterations to save the state of its computation. Upon receiving a signal, it reads this checkpoint file to resume from where it left off.
:p How does the application handle checkpoints?
??x
The application handles checkpoints by writing out a file named `checkpoint_name` every 60 iterations. This file contains the current iteration number. If the job is interrupted and restarted, the script picks up the last known state from this checkpoint file.
```c
if ( it % 60 == 0 ) {
    // Write out checkpoint file
}
```
x??

---

#### Submission of Restart Script
Background context: The batch script resubmits itself recursively until the job is completed. This process ensures that long-running jobs can continue even if they are interrupted.
:p How does the restart script handle its own submission?
??x
The restart script checks for a `DONE` file to indicate completion. If it detects this, it submits itself again with the same parameters to continue the job from where it left off.
```sh
if [ -z ${COUNT} ]; then
    export COUNT=0
fi

((COUNT++))
echo "Restart COUNT is $COUNT"

if [ . -e DONE ]; then
    if [ -e RESTART ]; then
        echo "=== Restarting $EXEC_NAME ===" >>$ OUTPUT_FILE
        cycle=`cat RESTART`
        rm -f RESTART
    else
        echo "=== Starting problem ===" >> $OUTPUT_FILE
        cycle=""
    fi

    mpirun -n ${NUM_CPUS}${EXEC_NAME}${cycle} &>>$ OUTPUT_FILE
    STATUS=$?
    echo "Finished mpirun" >> $OUTPUT_FILE

    if [ ${COUNT} -ge${MAX_RESTARTS} ]; then
        echo "=== Reached maximum number of restarts ===" >> $OUTPUT_FILE
        date > DONE
    fi

    if [ ${STATUS} = "0" -a . -e DONE ]; then
        echo "=== Submitting restart script ===" >> $OUTPUT_FILE
        sbatch <batch_restart.sh
    fi
fi
```
x??

---

#### Dependency Mechanism in Batch Scripts
Background context: In batch systems, specifying dependencies between jobs is crucial for managing job sequences and ensuring that certain jobs start only after their prerequisites have completed. This can be particularly useful for checkpoint-restart scenarios where a subsequent job needs to start based on the successful completion of a previous one.

If a job is submitted with `--dependency=afterok`, it will not start until the specified job has completed successfully (exit code 0). Here, `${SLURM_JOB_ID}` refers to the ID of the current job. The script checks for the existence of a `DONE` file and a `RESTART` file; if both exist, it restarts the application.

:p How does the batch script ensure that the next job starts only after the current one completes successfully?
??x
The script uses the dependency clause with `--dependency=afterok:${SLURM_JOB_ID}`. This means that the subsequent job will wait until the current job (`${SLURM_JOB_ID}`) has completed successfully before it starts.

```sh
sbatch --dependency=afterok:${SLURM_JOB_ID} batch_restart.sh
```
x??

---

#### Example Batch Script for Dependency
Background context: The provided script demonstrates how to submit a restart script that depends on the completion of the current job. This ensures higher priority in queueing, which can be beneficial depending on local scheduling policies.

:p What does the line `sbatch --dependency=afterok:${SLURM_JOB_ID}` do in the batch script?
??x
This command submits another batch script (`batch_restart.sh`) only after the current job has completed successfully (exit code 0). The `${SLURM_JOB_ID}` is a placeholder for the ID of the currently running job.

```sh
sbatch --dependency=afterok:${SLURM_JOB_ID} batch_restart.sh
```
x??

---

#### Handling Checkpoint/Restart Scenarios
Background context: In checkpoint-restart scenarios, it's common to have a script that handles both starting and restarting jobs. The script checks for the existence of a `DONE` file and a `RESTART` file. If these files exist, the script restarts the application; otherwise, it starts the application from scratch.

:p How does the batch script differentiate between a new start and a restart?
??x
The script differentiates by checking the presence of specific files:
- If both `DONE` and `RESTART` files exist, the job is in a restart state.
- Otherwise, the job is starting for the first time or has not completed successfully.

```sh
if [ . -e DONE ]; then
    if [ -e RESTART ]; then
        echo "=== Restarting ${EXEC_NAME} ===" >>${OUTPUT_FILE}
        cycle=`cat RESTART`
        rm -f RESTART
    else
        echo "=== Starting problem ===" >> ${OUTPUT_FILE}
        cycle=""
    fi
fi
```
x??

---

#### Submitting Jobs with Dependencies
Background context: The script uses the `--dependency=afterok` option to ensure that a job starts only after another specific job has completed successfully. This is crucial for managing workflow dependencies, especially in scenarios where jobs need to be sequential or conditional.

:p What is the purpose of using `--dependency=afterok:${SLURM_JOB_ID}` in the batch script?
??x
The purpose of using `--dependency=afterok:${SLURM_JOB_ID}` is to submit a subsequent job only after the current job (`${SLURM_JOB_ID}`) has completed successfully. This ensures that the next job does not start until the previous one is done, maintaining proper workflow order.

```sh
sbatch --dependency=afterok:${SLURM_JOB_ID} batch_restart.sh
```
x??

---

#### Multiple Restart Scenarios
Background context: The script handles multiple restart attempts by incrementing a counter and checking if it exceeds a maximum limit. If the limit is reached, it creates a `DONE` file to indicate that no further restarts are needed.

:p How does the script handle multiple restarts?
??x
The script uses an internal counter (`COUNT`) to keep track of the number of restarts. It increments this counter each time and checks if the count has reached or exceeded the maximum allowed restarts (`MAX_RESTARTS`). If so, it creates a `DONE` file to signal that no more restarts are needed.

```sh
if [ ${COUNT} -ge${MAX_RESTARTS} ]; then
    echo "=== Reached maximum number of restarts ===" >> ${OUTPUT_FILE}
    date > DONE
fi
```
x??

---

