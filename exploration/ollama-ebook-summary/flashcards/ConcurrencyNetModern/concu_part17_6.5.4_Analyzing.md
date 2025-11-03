# Flashcards: ConcurrencyNetModern_processed (Part 17)

**Starting Chapter:** 6.5.4 Analyzing tweet emotions using an Rx Pub-Sub class

---

---
#### RxPubSub Class Overview
The `RxPubSub` class is designed to facilitate event-driven programming using reactive extensions (Rx). It allows for subscription and unsubscription of observers, making it easier to handle asynchronous events. The class maintains state through private fields such as a `Subject`, an `observers` collection, and a `subscribed observables` list.

The primary constructor initializes the subject with either a default or specified one, ensuring that no direct manipulation can alter the internal state of the Subject. Additionally, it provides methods to manage subscriptions and unsubscriptions:

- **Dispose**: Invokes `OnCompleted` on the observer and removes it from the observers collection.
- **AsObservable**: Exposes the observable interface for applying high-order operations.

:p What is the purpose of the RxPubSub class?
??x
The purpose of the RxPubSub class is to provide a framework for implementing publish-subscribe pattern in C#, leveraging the power of reactive extensions (Rx). It allows for easy subscription and unsubscription of observers, managing their state through internal collections while ensuring encapsulation of the Subject.

```csharp
public class RxPubSub<T> : IDisposable
{
    private readonly Subject<T> subject;
    private readonly List<Subscription> observers = new List<Subscription>();
    private readonly List<IDisposable> observables = new List<IDisposable>();

    public RxPubSub(ISubject<T> subject) => this.subject = subject;

    // Other methods and properties...
}
```
x??

---
#### Subject State Management
In the `RxPubSub` class, the state of observers is managed through a private `observers` collection. Each observer has its own `Subscription`, which can be unsubscribed from by calling `Dispose`. This ensures that when an observer no longer needs to receive updates, it can be removed without affecting other observers.

The `subscribed observables` list maintains references to the interfaces implementing `IDisposable`, allowing for unregistration of these subscriptions as well.

:p How does the RxPubSub class manage the state of observers?
??x
The `RxPubSub` class manages the state of observers by maintaining a list of subscriptions (`observers`) and disposable observables. When an observer subscribes, it gets added to the `observers` collection with its corresponding `Subscription`. Similarly, when an observable is registered through the `AddPublisher` method, it's added to the `subscribed observables` list.

To unsubscribe, the `Dispose` method on a specific subscription or observable is called. This method invokes `OnCompleted` on the observer and removes it from the `observers` collection, ensuring that no more notifications are sent to it.

```csharp
public void Dispose()
{
    observer.OnCompleted();
    observers.Remove(observer);
}
```
x??

---
#### AsObservable Property Implementation
The `AsObservable` property in the `RxPubSub` class returns an observable interface exposed from the internal Subject. This is useful for applying high-order operations (like filtering, mapping) to event notifications without directly manipulating the Subject.

By exposing this property through `subject.AsObservable()`, developers can easily integrate and transform data streams without breaking encapsulation principles.

:p What does the AsObservable property do in the RxPubSub class?
??x
The `AsObservable` property in the `RxPubSub` class returns an observable interface exposed from the internal Subject. This allows for applying high-order operations such as filtering, mapping, or other transformations to the event notifications without directly accessing or modifying the internal state of the Subject.

This approach ensures that no one can perform an upper cast back to an `ISubject`, thereby maintaining encapsulation and preventing potential misuse or unintended changes to the Subject's state.

```csharp
public IObservable<T> AsObservable() => subject.AsObservable();
```
x??

---
#### Tweet Emotion Observable Implementation
In Listing 6.7, the `tweetEmotionObservable` function is implemented using the `Observable.Create` factory operator in F#. It creates an observable that processes a stream of tweet emotions by filtering, grouping, and mapping operations.

The function initializes Twitter credentials, starts the stream, filters tweets based on language, groups them by emotion, and maps each group to a `TweetEmotion` object before pushing it to the observer using `OnNext`.

:p How is the `tweetEmotionObservable` implemented in F#?
??x
In Listing 6.7, the `tweetEmotionObservable` function is implemented using the `Observable.Create` factory operator in F#. It creates an observable that processes a stream of tweet emotions by filtering, grouping, and mapping operations.

The implementation involves:

1. Initializing Twitter credentials.
2. Starting the stream with low-level filtering.
3. Filtering tweets to only include English language ones.
4. Grouping tweets by their evaluated emotion.
5. Mapping each group to a `TweetEmotion` object.
6. Pushing each processed tweet emotion to the observer using `OnNext`.

Here's the code snippet in detail:

```fsharp
let tweetEmotionObservable(throttle: TimeSpan) =
    Observable.Create(fun (observer: IObserver<_>) ->
        let cred = new TwitterCredentials(consumerKey, consumerSecretKey, accessToken, accessTokenSecret)
        let stream = Stream.CreateSampleStream(cred)

        stream.FilterLevel <- StreamFilterLevel.Low
        stream.StartStreamAsync() |> ignore

        stream.TweetReceived
        |> Observable.throttle(throttle) 
        |> Observable.filter(fun args -> 
            args.Tweet.Language = Language.English)
        |> Observable.groupBy(fun args ->
            evaluateEmotion args.Tweet.FullText)
        |> Observable.selectMany(fun args ->
            args |> Observable.map(fun tw ->
                TweetEmotion.Create tw.Tweet args.Key))
        |> Observable.subscribe(observer.OnNext)
    )
```
x??

---
#### C# Implementation of Tweet Emotion Observable
In Listing 6.8, the `tweetEmotionObservable` function is implemented in C# using similar logic to its F# counterpart but with a different syntax and library functions.

The implementation registers an event pattern for `TweetReceived`, applies throttling, filters tweets by language, groups them by emotion, and maps each group to a `TweetEmotion` object before pushing it to the observer.

:p How is the `tweetEmotionObservable` implemented in C#?
??x
In Listing 6.8, the `tweetEmotionObservable` function is implemented in C# using similar logic to its F# counterpart but with different syntax and library functions. It registers an event pattern for `TweetReceived`, applies throttling, filters tweets by language, groups them by emotion, and maps each group to a `TweetEmotion` object before pushing it to the observer.

Here's the code snippet in detail:

```csharp
var tweetObservable = Observable.FromEventPattern<TweetEventArgs>(stream, "TweetReceived");

Observable.Create<TweetEmotion>(observer =>
{
    var cred = new TwitterCredentials(consumerKey, consumerSecretKey, accessToken, accessTokenSecret);
    var stream = Stream.CreateSampleStream(cred);

    stream.FilterLevel = StreamFilterLevel.Low;
    stream.StartStreamAsync();

    return Observable.FromEventPattern<TweetReceivedEventArgs>(stream, "TweetReceived")
        .Throttle(throttle)
        .Select(args => args.EventArgs)
        .Where(args => args.Tweet.Language == Language.English)
        .GroupBy(args =>
            evaluateEmotion(args.Tweet.FullText))
        .SelectMany(args =>
            args.Select(tw => TweetEmotion.Create(tw.Tweet, args.Key)));
});
```
x??

---

#### Rx Publisher-Subscriber Pattern
Background context: The reactive extension (Rx) library for .NET allows developers to work with asynchronous sequences of data using a functional programming style. A publisher-subscriber pattern is used where an observable publisher emits events, and multiple subscribers can consume those events as needed.

:p What is the purpose of the Rx publisher-subscriber pattern described in the text?
??x
The purpose of the Rx publisher-subscriber pattern is to manage tweet emotion notifications by converting .NET CLR events into observable sequences. This allows for efficient handling and processing of asynchronous data streams, such as tweets, using higher-order operations like filtering, mapping, and reducing.

```csharp
// Example C# code showing how an event handler becomes an observable in F#
TweetReceived |> Observable.TweetEmotion
```
x??

---

#### TweetEmotion Struct Implementation
Background context: The `TweetEmotion` struct is used to store information about the tweet and its emotion. It provides properties for accessing these values, making it a lightweight data structure.

:p How is the `TweetEmotion` struct implemented in F#?
??x
The `TweetEmotion` struct is implemented with two members: `Tweet` and `Emotion`, along with a static method to create instances of this type. Here’s how:

```fsharp
[<Struct>]
type TweetEmotion(tweet:ITweet, emotion:Emotion) =
    member this.Tweet with get() = tweet
    member this.Emotion with get() = emotion

    static member Create tweet emotion = 
        TweetEmotion(tweet, emotion)
```

x??

---

#### RxTweetEmotion Class Implementation
Background context: The `RxTweetEmotion` class inherits from a base class and subscribes an observable to manage tweet emotion notifications. It uses the `FromEventPattern` method internally but does not require explicit creation of observables due to functional reactive programming (FRP) principles.

:p What is the role of the `RxTweetEmotion` class in handling tweet emotions?
??x
The role of the `RxTweetEmotion` class is to create and register an observable for tweet emotion notifications. It does this by subscribing to a predefined observable that processes tweets and their associated emotions, using a scheduler to handle concurrent messages.

```csharp
class RxTweetEmotion : RxPubSub<TweetEmotion>
{
    public RxTweetEmotion(TimeSpan throttle)
    {
        var obs = TweetsAnalysis.tweetEmotionObservable(throttle)
                                .SubscribeOn(TaskPoolScheduler.Default);

        base.AddPublisher(obs);
    }
}
```

x??

---

#### Implementing IObserver Interface
Background context: To respond to events in the observable stream, an observer interface (IObserver) needs to be implemented. Rx provides helper methods like `Observer.Create` to simplify this process.

:p How is the `TweetPositiveObserver` created and used?
??x
The `TweetPositiveObserver` is created using `Observer.Create`, which sets up a function to handle the `OnNext` event, printing positive tweets (those with happy emotions) to the console. Here’s how it can be implemented:

```csharp
IObserver<TweetEmotion> tweetPositiveObserver = 
    Observer.Create<TweetEmotion>(tweet => 
        if (tweet.Emotion.IsHappy)
            Console.WriteLine(tweet.Tweet.Text));
```

Then, this observer is subscribed to the `RxTweetEmotion` instance.

```csharp
IDisposable posTweets = rxTweetEmotion.Subscribe(tweetPositiveObserver);
```

x??

---

#### F# Object Expression for Observers
Background context: In F#, object expressions can be used to create instances of interfaces like `IObserver` on the fly, simplifying observer implementation. This approach is interoperable with C# and other .NET languages.

:p How does an F# object expression simplify implementing observers?
??x
An F# object expression provides a concise way to implement observer methods without defining named types. For example, to print only unhappy tweets:

```fsharp
let printUnhappyTweets() = 
    { new IObserver<TweetEmotion> with
        member this.OnNext(tweet) = 
            if tweet.Emotion = Unhappy then
                Console.WriteLine(tweet.Tweet.text)
        member this.OnCompleted() = ()
        member this.OnError(exn) = () }
```

This object expression can be used directly in C# code by referencing the F# library.

```csharp
IObserver<TweetEmotion> unhappyTweetObserver = printUnhappyTweets();
IDisposable disposable = rxTweetEmotion.Subscribe(unhappyTweetObserver);
```

x??

---

#### Task Parallelism Overview
Background context: Task parallelism is a paradigm where a program's execution is split into smaller tasks that can be executed concurrently. This approach aims to maximize processor utilization by distributing tasks across different processors, thereby reducing overall runtime.

:p What is task parallelism and how does it aim to reduce the total runtime?
??x
Task parallelism involves breaking down a computational problem into independent subtasks that can run in parallel on multiple threads or processors. By executing these tasks concurrently, the system can achieve better resource utilization and performance, ultimately reducing the overall execution time compared to sequential execution.

```java
// Pseudocode for initiating task parallelism using Java's ExecutorService
ExecutorService executor = Executors.newFixedThreadPool(numThreads);
for (Task task : tasks) {
    executor.submit(task);
}
executor.shutdown();
```
x??

---

#### Continuation-Passing Style (CPS)
Background context: Continuation-passing style (CPS) is a technique in functional programming where functions pass their continuations as arguments. This allows for the chaining of asynchronous operations and simplifies the handling of task parallelism by avoiding traditional locks.

:p What is continuation-passing style (CPS), and how does it help in task parallelism?
??x
Continuation-passing style (CPS) is a programming technique where functions are passed their continuations as arguments. In CPS, instead of returning values directly, functions return control to the caller (continuation), which can be used to continue execution after the function completes. This approach helps in task parallelism by eliminating the need for locks and enabling asynchronous processing.

```java
// Pseudocode for a simple CPS function
public void processTask(Task task, Continuation continuation) {
    // Perform some computation
    int result = compute(task);
    
    // Pass control to the continuation with the result
    continuation.execute(result);
}

interface Continuation {
    void execute(int result);
}
```
x??

---

#### Task Parallelism vs. Data Parallelism
Background context: Task parallelism and data parallelism are two distinct approaches in parallel computing. While task parallelism involves executing multiple independent tasks concurrently, data parallelism focuses on applying the same operation to different elements of a data set simultaneously.

:p How do task parallelism and data parallelism differ?
??x
Task parallelism involves running multiple independent tasks concurrently across processors. It breaks down a problem into smaller, independent subtasks that can be executed in parallel. Data parallelism, on the other hand, applies the same operation to different elements of a dataset simultaneously.

For example:
- **Task Parallelism**: Running multiple independent functions with shared starting data.
- **Data Parallelism**: Applying the same function across all elements of a data set.

```java
// Example of task parallelism
ExecutorService executor = Executors.newFixedThreadPool(numThreads);
List<Task> tasks = createTasks();
for (Task task : tasks) {
    executor.submit(task);
}
executor.shutdown();

// Example of data parallelism
List<Integer> numbers = Arrays.asList(1, 2, 3, 4, 5);
int[] results = new int[numbers.size()];
IntStream.range(0, numbers.size()).parallel().forEach(i -> results[i] = square(numbers.get(i)));
```
x??

---

#### Task-Based Functional Pipeline
Background context: Implementing a parallel functional pipeline involves composing multiple tasks that process data in sequence. Each task can run in parallel, and the output of one task is passed to another as input.

:p How does implementing a task-based functional pipeline work?
??x
Implementing a task-based functional pipeline involves breaking down a complex computation into smaller tasks that are executed in a pipelined manner. Tasks are composed using functional combinators, allowing each stage to run independently and in parallel. The output of one task serves as the input for the next.

Example:
1. **Task 1**: Computes square roots.
2. **Task 2**: Filters even numbers.
3. **Task 3**: Applies a transformation function.

```java
// Pseudocode for a simple functional pipeline
public void processPipeline(List<Integer> input, BiFunction<List<Integer>, Continuation, Void> pipeline) {
    List<Integer> squareRoots = computeSquareRoots(input);
    pipeline.apply(squareRoots, new Continuation() {
        @Override
        public void execute(List<Integer> results) {
            List<Integer> filteredEvenNumbers = filterEvenNumbers(results);
            pipeline.apply(filteredEvenNumbers, new Continuation() {
                @Override
                public void execute(List<Integer> finalResults) {
                    applyTransformation(finalResults);
                }
            });
        }
    });
}

// Task functions
List<Integer> computeSquareRoots(List<Integer> input) {
    return input.stream().map(Math::sqrt).collect(Collectors.toList());
}

List<Integer> filterEvenNumbers(List<Integer> input) {
    return input.stream().filter(n -> n % 2 == 0).collect(Collectors.toList());
}

void applyTransformation(List<Integer> input) {
    // Apply transformation logic
}
```
x??

---

#### Data Parallelism vs Task Parallelism
Background context explaining the differences between data and task parallelism. Data parallelism involves applying a single operation to many inputs, while task parallelism involves executing multiple diverse operations independently.

:p What is the main difference between data parallelism and task parallelism?
??x
Data parallelism applies a single operation to multiple inputs simultaneously, whereas task parallelism executes multiple independent tasks that may perform different operations on their own input. The key distinction lies in how the work is divided and executed.
x??

---
#### Task Parallelism in Real-World Scenarios
Explanation of why task parallelism is used in real-world scenarios where tasks are more complex and interdependent, making it challenging to split and reduce computations as easily as data.

:p Why is task parallelism preferred over data parallelism in some real-world applications?
??x
Task parallelism is preferred when dealing with complex, interconnected tasks that cannot be easily divided into independent jobs. It allows for the coordination of multiple functions running concurrently, which can handle dependencies between tasks and manage varying execution times.
x??

---
#### Why Use Functional Programming (FP) with Task Parallelism?
Explanation on how functional programming (FP) aids in task parallelism by providing tools to control side effects and manage task dependencies.

:p How does functional programming help with task parallelism?
??x
Functional programming helps with task parallelism by promoting the use of pure functions, which are free from side effects. This leads to referential transparency and deterministic code, making it easier to reason about tasks running in parallel. Functional concepts like immutability also simplify managing shared state.
x??

---
#### Pure Functions in Task Parallelism
Explanation on how pure functions contribute to the effectiveness and predictability of task-based parallel programs.

:p Why are pure functions important for task parallelism?
??x
Pure functions are crucial because they always produce the same output given the same input, regardless of external state. This makes them ideal for parallel execution since their order of execution is irrelevant. Pure functions ensure that tasks can run independently without affecting each other's results.
x??

---
#### Side Effects in Task Parallelism
Explanation on how to handle side effects when using task parallelism.

:p How should side effects be managed in task parallelism?
??x
Side effects should be controlled locally by performing computations within a function isolated from external state. To avoid conflicts, defensive copying can be used to create immutable copies of mutable objects that can be safely shared without affecting the original.
x??

---
#### Immutable Structures in Task Parallelism
Explanation on why using immutable structures is beneficial when tasks must share data.

:p Why use immutable structures in task parallelism?
??x
Immutable structures are beneficial because they prevent unintended modifications from one task affecting others. By ensuring that once a value is created, it cannot be changed, shared state issues are minimized, and the program becomes more predictable.
x??

---
#### Defensive Copy Approach
Explanation of defensive copying as a mechanism to manage mutable objects in parallel tasks.

:p What is defensive copying?
??x
Defensive copying is a technique used to create a copy of an object that can be safely shared among tasks. This prevents modifications from one task from affecting the original object, thus managing side effects and ensuring data integrity.
x??

---
#### Example of Defensive Copying in Code
Example of defensive copying code with explanation.

:p Provide an example of defensive copying in C# or Java.
??x
Here is an example of defensive copying in Java:

```java
public class Example {
    private String mutableData;

    public Example(String data) {
        this.mutableData = new StringBuilder(data).toString(); // Defensive copy
    }

    // Getter and other methods...
}
```

In this example, a defensive copy of the mutable string is created during initialization. This ensures that any modifications made to `mutableData` within tasks do not affect the original input.
x??

---

---
#### ThreadPool Class Overview
Background context explaining how `ThreadPool` works and its benefits. The `.NET Framework` provides a static class called `ThreadPool` which optimizes performance by reusing existing threads instead of creating new ones, thus minimizing overhead.

:p What is the primary advantage of using the `ThreadPool` class in multithreading?
??x
The primary advantage of using the `ThreadPool` class is that it optimizes performance and reduces memory consumption by reusing existing threads. This approach minimizes the overhead associated with thread creation and destruction, making your application more efficient.

```csharp
// Example usage of ThreadPool.QueueUserWorkItem
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.nasdaq.com"));
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.bbc.com"));
```
x??

---
#### Comparison with Conventional Thread Creation
Background context comparing conventional thread creation to `ThreadPool` usage. In conventional multithreading, each task requires the instantiation of a new thread, leading to potential memory consumption issues and increased overhead.

:p What is the main difference between creating threads using `Thread` class and using `ThreadPool` in .NET?
??x
The main difference lies in resource management and efficiency. When you create threads using the `Thread` class, each task requires its own thread, which can lead to memory consumption issues due to large stack sizes and context switches. In contrast, `ThreadPool` reuses existing threads to execute tasks, minimizing overhead.

```csharp
// Conventional thread creation example
var threadA = new Thread(() => downloadSite("http://www.nasdaq.com"));
var threadB = new Thread(() => downloadSite("http://www.bbc.com"));
threadA.Start();
threadB.Start();
threadA.Join();
threadB.Join();
```
x??

---
#### QueueUserWorkItem Method
Background context explaining the `QueueUserWorkItem` method. This static method allows you to queue tasks for execution by the `ThreadPool`, providing a lightweight way to manage tasks without explicitly creating threads.

:p How does the `QueueUserWorkItem` method facilitate task management in .NET?
??x
The `QueueUserWorkItem` method facilitates task management by allowing you to queue tasks for execution by the `ThreadPool`. This approach minimizes overhead since the `ThreadPool` reuses existing threads, avoiding the need for frequent thread creation and destruction.

```csharp
// Example usage of QueueUserWorkItem
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.nasdaq.com"));
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.bbc.com"));
```
x??

---
#### Conventional Thread vs. ThreadPool Performance
Background context discussing the performance implications of using conventional threads versus `ThreadPool`. Conventional thread creation is expensive due to overhead and memory usage, while `ThreadPool` optimizes performance by reusing existing threads.

:p What are the performance benefits of using `ThreadPool` over creating new threads explicitly?
??x
The performance benefits of using `ThreadPool` include reduced overhead, minimized memory consumption, and efficient reuse of existing threads. This approach is more resource-friendly compared to creating new threads for each task, which can lead to higher memory usage and increased context switching.

```csharp
// Conventional thread creation example
var downloadSite = url => { 
    var content = new WebClient().DownloadString(url); 
    Console.WriteLine($"The size of the web site {url} is {content.Length}");
};

var threadA = new Thread(() => downloadSite("http://www.nasdaq.com"));
var threadB = new Thread(() => downloadSite("http://www.bbc.com"));

threadA.Start();
threadB.Start();

threadA.Join();
threadB.Join();
```
x??

---
#### ThreadPool and Task Scheduling
Background context on how `ThreadPool` schedules tasks. The `ThreadPool` schedules tasks by reusing threads for the next available work item, returning them to the pool once completed.

:p How does the `ThreadPool` manage task scheduling?
??x
The `ThreadPool` manages task scheduling by reusing existing threads for new work items as they become available. Once a thread completes its current task, it is returned to the pool to handle another task, thus optimizing resource usage and reducing overhead.

```csharp
// Example of ThreadPool task scheduling
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.nasdaq.com"));
ThreadPool.QueueUserWorkItem(o => downloadSite("http://www.bbc.com"));
```
x??

---

