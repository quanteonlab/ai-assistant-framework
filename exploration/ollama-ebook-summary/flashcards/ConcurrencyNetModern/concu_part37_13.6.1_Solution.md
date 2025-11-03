# Flashcards: ConcurrencyNetModern_processed (Part 37)

**Starting Chapter:** 13.6.1 Solution implementing a polymorphic  publisher-subscriber pattern

---

#### ReaderWriterAgent Pattern for Concurrency
Background context: The `ReaderWriterAgent` pattern is a design approach used to manage concurrent access to shared resources. It ensures that writes are serialized and non-blocking, while allowing multiple readers simultaneously. This is particularly useful when different threads need to read from or write to a mutable dictionary like `myDB`.

In the provided example, an agent is created using `ReaderWriterAgent` which wraps around a `MailboxProcessor`. The `agentSql` function defines how messages are handled (reads and writes) on this mailbox. The pattern helps in managing concurrency without blocking operations.

:p How does the `ReaderWriterAgent` ensure thread-safe access to a shared resource?
??x
The `ReaderWriterAgent` ensures thread-safe access by serializing write operations while allowing multiple read operations concurrently. This is achieved through a combination of a mailbox processor that handles requests (reads and writes) in an asynchronous manner, ensuring that no two write operations can occur simultaneously.

In the example, the `agentSql` function defines how messages are processed:
- **Read** operation: Retrieves data from the dictionary.
- **Write** operation: Adds new entries to the dictionary.

This pattern prevents race conditions and ensures that concurrent reads do not interfere with writes, but writes block until they complete.

```fsharp
let agentSql connectionString = 
    fun (inbox: MailboxProcessor<_>) ->        
        let rec loop() = async {            
            let. msg = inbox.Receive()
            match msg with
            | Read(Get(id, reply)) ->
                match myDB.TryGetValue(id) with
                | true, res -> reply.Reply(Some res)
                | _ -> reply.Reply(None)
            | Write(Add(person, reply)) ->
                let id = myDB.Count 
                myDB.Add(id, {person with id = id}) 
                reply.Reply(Some id)            
            return. loop()        
        }
        loop()
```
x??

---

#### Thread-safe Random Number Generator
Background context: In multithreaded applications, generating random numbers can be a challenge due to the shared state of `System.Random`. This class may not provide thread safety when accessed concurrently by multiple threads. To address this, `ThreadLocal<T>` is used to ensure each thread has its own instance of `Random`.

The example provided shows how to create a thread-safe pseudo-random number generator using `ThreadLocal<Random>`. The `ThreadSafeRandom` class inherits from `Random` and overrides the necessary methods.

:p How does the `ThreadSafeRandom` class ensure that random numbers are generated safely in a multi-threaded environment?
??x
The `ThreadSafeRandom` class ensures thread safety by leveraging `ThreadLocal<T>` to create an isolated instance of `Random` for each thread. This prevents shared state issues and guarantees that each thread gets independent random number generation.

Here is how the `ThreadSafeRandom` class works:

```csharp
public class ThreadSafeRandom : Random {
    private ThreadLocal<Random> random = 
        new ThreadLocal<Random>(() => new Random(MakeRandomSeed()));

    public override int Next() => random.Value.Next();
    public override int Next(int maxValue) => random.Value.Next(maxValue);
    public override int Next(int minValue, int maxValue) =>
        random.Value.Next(minValue, maxValue);
    public override double NextDouble() => random.Value.NextDouble();
    public override void NextBytes(byte[] buffer) =>
        random.Value.NextBytes(buffer);

    static int MakeRandomSeed() => 
        Guid.NewGuid().ToString().GetHashCode();
}
```

The `MakeRandomSeed` method generates a unique seed for each thread, ensuring that the state of `Random` is different across threads. This setup allows multiple threads to generate independent sequences of random numbers without interfering with each other.

x??

---

---
#### ThreadLocal Random Number Generation
Background context: In concurrent programming, generating unique and independent random numbers for each thread is crucial to avoid race conditions. The `ThreadLocal<T>` class allows creating a local copy of an object per thread, ensuring thread safety without blocking other threads.

Code example:
```csharp
var safeRandom = new ThreadSafeRandom();
string[] clips = new string[] { "1.mp3", "2.mp3", "3.mp3", "4.mp3" };
Parallel.For(0, 1000, (i) =>
{
    var clipIndex = safeRandom.Next(4);
    var clip = clips[clipIndex];
    Console.WriteLine($"clip to play {clip} - Thread Id {Thread.CurrentThread.ManagedThreadId}");
});
```
:p How does `ThreadLocal<T>` ensure thread safety in generating random numbers?
??x
`ThreadLocal<T>` ensures thread safety by creating a separate instance of the `Random` class for each thread, which are not dependent on the system clock. This way, each thread has its own unique and independent random number generator.

```csharp
public sealed class ThreadSafeRandom : Random
{
    private static readonly ThreadLocal<Random> ThreadLocalInstance = new(() => new Random());

    public override int Next()
    {
        return ThreadLocalInstance.Value.Next();
    }
}
```
x?
---

---
#### Polymorphic Event Aggregator Design Pattern
Background context: The `EventAggregator` pattern is a design pattern used to manage events of different types in a loosely coupled manner. It allows publishers to raise events and subscribers to handle them without knowing the specific event type.

Code example:
```csharp
let evtAggregator = EventAggregator.Create()
type IncrementEvent = { Value: int }
type ResetEvent = { ResetTime: DateTime }

evtAggregator.GetEvent<ResetEvent>()
    .ObserveOn(Scheduler.CurrentThread)
    .Subscribe(fun evt -> 
        printfn "Counter Reset at: %O - Thread Id %d" evt.ResetTime Thread.CurrentThread.ManagedThreadId)

evtAggregator.GetEvent<IncrementEvent>()
    .ObserveOn(Scheduler.CurrentThread)
    .Subscribe(fun evt -> 
        printfn "Counter Incremented. Value: %d - Thread Id %d" evt.Value Thread.CurrentThread.ManagedThreadId)

for i in [0..10] do
    evtAggregator.Publish({ Value = i })

evtAggregator.Publish({ ResetTime = DateTime(2015, 10, 21) })
```
:p How does the `EventAggregator` manage different types of events?
??x
The `EventAggregator` manages different types of events by using Rx (Reactive Extensions) subjects. Each event type is treated as an observable sequence, and when an event is published, it notifies all subscribers that are interested in that specific event type.

```csharp
type IEventAggregator = 
    inherit IDisposable
    abstract GetEvent<'Event> : unit -> IObservable<'Event>
    abstract Publish<'Event> : eventToPublish: 'Event -> unit

type internal EventAggregator() =
    let disposedErrorMessage = "The EventAggregator is already disposed."
    let subject = new Subject<obj>()

    interface IEventAggregator with
        member this.GetEvent<'Event>() : IObservable<'Event> =
            if (subject.IsDisposed) then failwith disposedErrorMessage
            subject.OfType<'Event>().AsObservable<'Event>()
                .SubscribeOn(TaskPoolScheduler.Default)

        member this.Publish(eventToPublish: 'Event): unit = 
            if (subject.IsDisposed) then failwith disposedErrorMessage
            subject.OnNext(eventToPublish)

        member this.Dispose(): unit = subject.Dispose()

static member Create() : IEventAggregator = new EventAggregator()
```
x?

#### Custom Rx Scheduler for Controlling Degree of Parallelism
Background context explaining the need to control parallelism in Rx when dealing with large volumes of event streams. The default schedulers provided by Rx, like `TaskPool` and `ThreadPool`, may not be optimal due to their initial thread creation delays.

:p What is the main issue with the default Rx schedulers for controlling degree of parallelism?
??x
The main issue is that both the TaskPool and ThreadPool schedulers start with one thread and only increase the number of threads after a 500ms delay. This can lead to suboptimal performance, especially on multi-core systems where actions are queued unnecessarily.

```fsharp
// Example code snippet showing default Rx scheduler behavior
let subscribeOnDefault = Observable.Interval(TimeSpan.FromSeconds(0.4))
                                     .SubscribeOn(ThreadPoolScheduler.Instance)
                                     .Subscribe(fun _ -> 
                                         printfn "ThreadId: %d" Thread.CurrentThread.ManagedThreadId)
```
x??

---

#### ParallelAgentScheduler Implementation
The custom `ParallelAgentScheduler` is implemented to provide finer control over the degree of parallelism in Rx. It uses a pool of agent workers to manage notifications and ensure that a specified number of tasks can run concurrently without unnecessary delays.

:p How does the `ParallelAgentScheduler` handle job scheduling?
??x
The `ParallelAgentScheduler` handles job scheduling by using an internal priority queue managed by a `MailboxProcessor`. When jobs are received, they are inserted into this queue. The scheduler then executes these jobs in parallel based on their due times and prioritization.

```fsharp
// Implementation of the agent scheduler logic
let schedulerAgent (inbox:MailboxProcessor<ScheduleMsg>) =
    let rec execute (queue:IPriorityQueue<ScheduleRequest>) = 
        async { ... }
    // Other parts of the implementation ...
```
x??

---

#### Custom Scheduler Interface for Rx
The `ParallelAgentScheduler` implements the `IScheduler` interface to integrate with Rx. This allows it to be used as a valid scheduler in Rx operations like `SubscribeOn`.

:p What does the `IScheduler` interface allow you to implement?
??x
The `IScheduler` interface allows you to define custom scheduling logic for Rx, enabling features such as `SubscribeOn`, `ObserveOn`, and other time-based operators. Implementing this interface provides control over when actions are executed based on the defined scheduler.

```fsharp
// Example of using ParallelAgentScheduler with SubscribeOn in F#
let scheduler = ParallelAgentScheduler(4)
Observable.Interval(TimeSpan.FromSeconds(0.4))
           .SubscribeOn(scheduler)
           .Subscribe(fun _ -> 
               printfn "ThreadId: %d" Thread.CurrentThread.ManagedThreadId)
```
x??

---

#### Scheduling Jobs in ParallelAgentScheduler
The `ParallelAgentScheduler` uses a custom message type and asynchronous processing to manage the scheduling of jobs. It ensures that tasks are executed based on their priority, avoiding unnecessary delays.

:p How does the `ParallelAgentScheduler` handle job execution?
??x
The `ParallelAgentScheduler` handles job execution by using an agent-based approach with a priority queue. When a job request arrives, it is inserted into the priority queue and scheduled for execution as soon as possible based on its priority. This ensures that tasks are processed in a controlled manner without unnecessary delays.

```fsharp
// Pseudocode of how jobs are handled in ParallelAgentScheduler
let execute (queue:IPriorityQueue<ScheduleRequest>) = 
    async {
        // Code to pop and process jobs from the queue
        match queue |> PriorityQueue.tryPop with
            | Some(req, tail) -> 
                let timeout = int (req.Due - DateTimeOffset.Now).TotalMilliseconds
                if timeout > 0 && not req.IsCanceled then 
                    return idle queue timeout
                else
                    // Execute action if it's not canceled
                    req.Action.Invoke()
                    return execute tail
            | None -> return idle queue -1
    }
```
x??

---

#### Custom Scheduler for Concurrent Programming with Rx
The `ParallelAgentScheduler` is a custom Rx scheduler that supports concurrent programming by managing the degree of parallelism. It uses an agent to coordinate and manage notifications, ensuring that tasks are executed in parallel without unnecessary delays.

:p What makes `ParallelAgentScheduler` different from default Rx schedulers?
??x
What makes `ParallelAgentScheduler` different from default Rx schedulers is its ability to provide fine-grained control over the degree of parallelism. Unlike the default schedulers which start with a single thread and only increase concurrency after a delay, `ParallelAgentScheduler` initializes its internal thread pool at startup time, reducing initial delays and improving performance on multi-core systems.

```fsharp
// Example of configuring ParallelAgentScheduler
let scheduler = ParallelAgentScheduler(4)
```
x??

---

#### Managing Concurrency with Rx
By implementing a custom scheduler like `ParallelAgentScheduler`, you can manage concurrency more effectively in Rx-based applications. This is particularly useful for large volumes of event streams where precise control over parallelism can significantly impact performance.

:p How does managing concurrency help in Rx-based applications?
??x
Managing concurrency helps in Rx-based applications by ensuring that tasks are executed efficiently without overwhelming the system resources. By controlling the degree of parallelism, you can optimize performance, reduce delays, and ensure that your application handles large volumes of data more effectively.

```fsharp
// Example usage of custom scheduler in an Rx pipeline
let scheduler = ParallelAgentScheduler(4)
Observable.Interval(TimeSpan.FromSeconds(0.4))
           .SubscribeOn(scheduler)
           .Subscribe(fun _ -> 
               printfn "ThreadId: %d" Thread.CurrentThread.ManagedThreadId)
```
x??

