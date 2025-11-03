# Flashcards: ConcurrencyNetModern_processed (Part 21)

**Starting Chapter:** 8.5 TAP a case study

---

#### Task-Based Asynchronous Programming (TAP) Overview
Background context: The provided text discusses implementing asynchronous programming using C#. TAP, or Task-based Asynchronous Pattern, allows for efficient handling of I/O-bound and CPU-bound operations without blocking the UI thread. It provides a rich set of tools to manage asynchronous tasks effectively.
:p What is TAP, and why is it useful in handling network operations?
??x
TAP, or Task-Based Asynchronous Programming, is a pattern that enables developers to write asynchronous code more easily by leveraging `Task` objects and `async/await`. It's particularly useful for I/O-bound tasks like downloading data from the internet, where using synchronous methods would block the UI thread, making the application unresponsive. 
??x
```csharp
async Task<Bitmap> DownloadAndSaveFaviconAsync(string domain, string fileDestination)
{
    // Code from the provided text
}
```
This example demonstrates how TAP can be used to download a favicon asynchronously and save it to disk without blocking.
x??

---

#### Asynchronous Method Example: Downloading Favicon
Background context: The code snippet shows an asynchronous method that downloads a favicon using HTTP GET requests and then processes it to save as a bitmap file. It uses `HttpClient` for making the HTTP request and `Bitmap.FromStream` for processing the image data.
:p How does the provided code demonstrate TAP in action?
??x
The provided code demonstrates how TAP can be used to handle asynchronous operations efficiently. By using `async/await`, it ensures that the application remains responsive while performing network requests. The use of `HttpClient` and `Bitmap.FromStream` are key parts of this process.
??x
```csharp
async Task<Bitmap> DownloadAndSaveFaviconAsync(string domain, string fileDestination)
{
    using (var httpClient = new HttpClient())
    {
        var response = await httpClient.GetAsync($"http://{domain}/favicon.ico");
        byte[] bytes = await response.Content.ReadAsByteArrayAsync();
        return Bitmap.FromStream(new MemoryStream(bytes));
    }
}
```
This code snippet shows how to asynchronously download a favicon and convert it into a bitmap. The `async` keyword indicates that the method can perform asynchronous operations, while `await` is used to wait for those operations without blocking.
x??

---

#### Parallel Processing with TAP
Background context: The text explains how parallel processing can be achieved using TAP in C#. It mentions the use of `Select` and `Tap` (a function composition library) to process multiple tasks concurrently, optimizing performance by leveraging all available cores.
:p How does TAP facilitate parallel processing?
??x
TAP facilitates parallel processing by allowing developers to write asynchronous code that can be easily composed and executed in parallel. The `Select` method is used to transform each item in a sequence into a new task, and `Tap` (or `SelectMany` with `async` methods) ensures these tasks are executed concurrently.
??x
```csharp
var stockHistoryTasks = stockSymbols.Select(stock => ProcessStockHistory(stock));
var stockHistories = await Task.WhenAll(stockHistoryTasks);
```
This code demonstrates how to use `Select` and `Task.WhenAll` to process multiple stocks in parallel, ensuring that the tasks run concurrently.
x??

---

#### Stock Data Analysis with TAP
Background context: The example provided shows an implementation of a program that downloads historical stock data from Google Finance, processes it into `StockData` objects, and then analyzes the results. This is done asynchronously to improve performance and responsiveness.
:p How does the given code achieve parallel processing for downloading and analyzing multiple stocks?
??x
The code achieves parallel processing by using LINQ's `Select` method to transform each stock symbol into a task that processes its historical data asynchronously. The `Task.WhenAll` method then waits for all these tasks to complete, allowing them to run in parallel.
??x
```csharp
var stockHistoryTasks = stockSymbols.Select(stock => ProcessStockHistory(stock));
await Task.WhenAll(stockHistoryTasks);
```
This line of code initiates the asynchronous processing of multiple stocks by creating a `Task` for each symbol and waiting for all of them to complete using `Task.WhenAll`.
x??

---

#### Downloading Historical Stock Data
Background context: The method `DownloadStockHistory` demonstrates how to asynchronously download historical stock data from an HTTP endpoint. It uses `HttpClient` to make the request and processes the CSV response.
:p What is the role of `HttpClient` in this example?
??x
The role of `HttpClient` in this example is to manage asynchronous HTTP requests, allowing for non-blocking operations that do not freeze the UI thread. It provides a straightforward way to handle GET requests to retrieve historical stock data.
??x
```csharp
async Task<string> DownloadStockHistory(string symbol)
{
    string url = $"http://www.google.com/finance/historical?q={symbol}&output=csv";
    var request = WebRequest.Create(url);
    using (var response = await request.GetResponseAsync().ConfigureAwait(false))
    using (var reader = new StreamReader(response.GetResponseStream()))
        return await reader.ReadToEndAsync().ConfigureAwait(false);
}
```
This code snippet shows how `HttpClient` can be used to asynchronously download historical stock data. The method makes an HTTP GET request, retrieves the response, and reads it as a string.
x??

---

#### Analyzing Stock Data
Background context: The `AnalyzeStockHistory` function orchestrates downloading and analyzing multiple stocks' historical data in parallel using TAP. It measures the time taken for processing.
:p How does `AnalyzeStockHistory` optimize program execution?
??x
`AnalyzeStockHistory` optimizes program execution by leveraging TAP to process multiple stock symbols concurrently. By using `Select`, it transforms each symbol into a task, and then `Task.WhenAll` ensures that all tasks run in parallel, improving overall performance.
??x
```csharp
async Task AnalyzeStockHistory(string[] stockSymbols)
{
    var sw = Stopwatch.StartNew();
    IEnumerable<Task<Tuple<string, StockData[]>>> stockHistoryTasks = 
        stockSymbols.Select(stock => ProcessStockHistory(stock));
    var stockHistories = new List<Tuple<string, StockData[]>>();
    foreach (var stockTask in stockHistoryTasks) 
        stockHistories.Add(await stockTask);
    ShowChart(stockHistories, sw.ElapsedMilliseconds);
}
```
This code snippet shows how `AnalyzeStockHistory` uses TAP to process multiple stocks' data concurrently. It starts a stopwatch, processes each symbol into a task, and then waits for all tasks to complete before rendering the chart.
x??

---

#### Asynchronous I/O Operations Using TAP Pattern
Background context: The provided text discusses how to perform asynchronous operations for downloading and processing stock history using the Task-based Asynchronous Pattern (TAP) in C#. The code uses `GetResponseAsync()` and `ReadToEndAsync()` methods to handle long-running I/O operations asynchronously. The data is then parsed into a structured format, such as an object of type `StockData`, utilizing LINQ and running computationally intensive tasks on the ThreadPool.

:p What are the key methods used for performing asynchronous I/O operations in this context?
??x
The key methods used are `GetResponseAsync()` and `ReadToEndAsync()`. These methods perform I/O operations asynchronously, allowing the application to remain responsive while waiting for data or a response from an external source such as a web service.

```csharp
var response = await httpClient.GetAsync("https://finance.google.com/stockquote/<symbol>");
string content = await response.Content.ReadAsStringAsync();
```
x??

---

#### Parsed Data into Structured Format
Background context: After reading the CSV data asynchronously, it is parsed using LINQ and a function `ConvertStockHistory()` to transform the raw data into an understandable structure. The transformation runs on the ThreadPool using `Task.Run()`, ensuring that CPU-intensive tasks are offloaded from the main thread.

:p How does the `ConvertStockHistory()` function handle the data transformation?
??x
The `ConvertStockHistory()` function uses `Task.Run()` to run a lambda expression on the ThreadPool, which performs the data transformation. This approach ensures that computationally intensive operations do not block the UI or other critical threads.

```csharp
var stockData = await Task.Run(() => ConvertStockHistory(csvContent));
```
x??

---

#### Asynchronous Method Return Type and Task Handling
Background context: The method `ProcessStockHistory()` is marked as `async` to handle asynchronous operations, returning a `Task<Tuple<string, StockData[]>>`. This return type indicates that the method performs an asynchronous operation and ultimately returns a value.

:p What is the return type of the `ProcessStockHistory()` method?
??x
The return type of the `ProcessStockHistory()` method is `Task<Tuple<string, StockData[]>>`. This means the method performs an asynchronous operation and eventually returns a tuple containing string and an array of `StockData` objects.

```csharp
public async Task<Tuple<string, StockData[]>> ProcessStockHistory()
{
    // Asynchronous operations here...
    return Tuple.Create("Identifier", await GetStockDataAsync());
}
```
x??

---

#### Lazy Collection of Asynchronous Operations
Background context: The method processes the historical data asynchronously and one at a time. This approach ensures that each operation is handled efficiently without overwhelming the system resources.

:p How does the code ensure efficient processing of asynchronous operations?
??x
The code ensures efficient processing by using `Task.Run()` to offload computationally intensive tasks, such as parsing CSV data or converting it into a structured format, onto the ThreadPool. This approach prevents blocking the main thread and allows for better resource utilization.

```csharp
var convertedData = await Task.Run(() => ConvertStockHistory(csvContent));
```
x??

---

#### Web Request to Retrieve Stock History
Background context: The example uses a web request to retrieve stock history from Google Finance using an endpoint. This method is asynchronous, allowing the application to handle other tasks while waiting for the response.

:p How does the code make the web request to fetch stock history?
??x
The code makes the web request to fetch stock history asynchronously using `HttpClient` and its methods `GetAsync()` and `ReadAsStringAsync()`. These methods are marked with the async keyword, ensuring that they handle I/O operations efficiently without blocking the UI thread.

```csharp
var response = await httpClient.GetAsync("https://finance.google.com/stockquote/<symbol>");
string content = await response.Content.ReadAsStringAsync();
```
x??

---

#### Chart Display and Time Elapsed
Background context: After processing the stock history, the data is sent to a method `ShowChart()` to display the chart and record the elapsed time. This ensures that the UI remains responsive during asynchronous operations.

:p What does the `ShowChart()` method do?
??x
The `ShowChart()` method displays the processed stock history on a chart and records the elapsed time for the entire process, ensuring that the user interface remains responsive during asynchronous operations.

```csharp
public void ShowChart(Tuple<string, StockData[]> data)
{
    // Code to display the chart and record elapsed time
}
```
x??

---

#### High-Level Architecture of a Scalable Service
Background context: The example demonstrates how an application can handle multiple concurrent requests efficiently by leveraging asynchronous programming. This architecture is scalable as it can optimize local hardware resources, reducing thread usage and maintaining system responsiveness.

:p What are the key steps in processing stock history asynchronously?
??x
The key steps in processing stock history asynchronously include:
1. Making an asynchronous web request to retrieve the data.
2. Reading the CSV content asynchronously.
3. Parsing the data using `ConvertStockHistory()` and running computationally intensive tasks on the ThreadPool.
4. Returning the processed data as a `Task<Tuple<string, StockData[]>>`.

```csharp
public async Task<Tuple<string, StockData[]>> ProcessStockHistory()
{
    var response = await httpClient.GetAsync("https://finance.google.com/stockquote/<symbol>");
    string content = await response.Content.ReadAsStringAsync();
    var stockData = await Task.Run(() => ConvertStockHistory(content));
    return Tuple.Create("Identifier", stockData);
}
```
x??

---

#### Asynchronous Programming Model for Downloading Data
Asynchronous programming allows operations to run concurrently, improving overall performance. In this model, database queries and network requests are processed without blocking threads, allowing other tasks to proceed simultaneously.
:p What is the asynchronous approach used for downloading data from a stock market service?
??x
The asynchronous approach ensures that database queries and network operations do not block threads, enabling parallel execution. This leads to improved responsiveness and efficiency in applications.
```csharp
async Task<string> DownloadStockHistory(string symbol, CancellationToken token)
{
    string stockUrl = $"http://www.google.com/finance/historical?q={symbol}&output=csv";
    var request = await new HttpClient().GetAsync(stockUrl, token);
    return await request.Content.ReadAsStringAsync();
}
```
x??

---

#### Asynchronous Cancellation
In asynchronous operations, it is often necessary to terminate long-running tasks prematurely. The .NET Framework provides a cooperative cancellation mechanism using `CancellationToken` and `CancellationTokenSource`.
:p How can you cancel an asynchronous operation in the .NET Framework?
??x
To cancel an asynchronous operation, use `CancellationToken` with `CancellationTokenSource`. When cancellation is requested, any active operations will throw an `OperationCanceledException`, allowing graceful termination.
```csharp
CancellationTokenSource cts = new CancellationTokenSource();
async Task<string> DownloadStockHistory(string symbol, CancellationToken token)
{
    string stockUrl = $"http://www.google.com/finance/historical?q={symbol}&output=csv";
    var request = await new HttpClient().GetAsync(stockUrl, token);
    return await request.Content.ReadAsStringAsync();
}
cts.Cancel();
```
x??

---

#### Cancellation Token in Asynchronous Methods
Methods that support cancellation use `CancellationToken` to monitor the state of a task. The `ThrowIfCancellationRequested()` method can be used to stop execution if cancellation is pending.
:p How do you integrate cancellation support into an existing asynchronous method?
??x
To integrate cancellation, pass a `CancellationToken` as a parameter and regularly check its state using `ThrowIfCancellationRequested()`. If the token indicates cancellation, the operation should terminate without returning a result.
```csharp
List<Task<Tuple<string, StockData[]>>> stockHistoryTasks = 
    stockSymbols.Select(async symbol => {
        var url = $"http://www.google.com/finance/historical?q={symbol}&output=csv";
        var request = HttpWebRequest.Create(url);
        using (var response = await request.GetResponseAsync())
            using (var reader = new StreamReader(response.GetResponseStream())) {
                token.ThrowIfCancellationRequested();
                var csvData = await reader.ReadToEndAsync();
                var prices = await ConvertStockHistory(csvData);
                token.ThrowIfCancellationRequested();
                return Tuple.Create(symbol, prices.ToArray());
            }
    }).ToList();
```
x??

---

#### Registering Callbacks for Cancellation
You can register a callback to execute when a cancellation is requested. This allows for custom actions such as logging or notifying listeners.
:p How do you use the `Register` method with a `CancellationToken`?
??x
The `Register` method on `CancellationToken` allows you to specify an action that will be executed if cancellation is requested. In this example, it cancels a `WebClient` instance's download operation.
```csharp
CancellationTokenSource tokenSource = new CancellationTokenSource();
CancellationToken token = tokenSource.Token;
Task.Run(async () => {
    var webClient = new WebClient();
    token.Register(() => webClient.CancelAsync());
    var data = await webClient.DownloadDataTaskAsync("http://www.manning.com");
}, token);
tokenSource.Cancel();
```
x??

---

#### Cooperative Cancellation with Linked Tokens
Cooperative cancellation can be applied to multiple reasons by creating a composite `CancellationToken`. This is useful when combining multiple cancellation sources.
:p How do you create and use a composite `CancellationToken`?
??x
To create a composite `CancellationToken`, use `CreateLinkedTokenSource()` which generates a token that will be canceled if any of the specified tokens are canceled. In this example, a composite token is used to cancel operations from two different cancellation sources.
```csharp
CancellationTokenSource ctsOne = new CancellationTokenSource();
CancellationTokenSource ctsTwo = new CancellationTokenSource();
CancellationTokenSource ctsComposite = CancellationTokenSource.CreateLinkedTokenSource(ctsOne.Token, ctsTwo.Token);
CancellationToken ctsCompositeToken = ctsComposite.Token;
Task.Factory.StartNew(async () => {
    var webClient = new WebClient();
    ctsCompositeToken.Register(() => webClient.CancelAsync());
    var data = await webClient.DownloadDataTaskAsync("http://www.manning.com");
}, ctsComposite.Token);
ctsComposite.Cancel();
```
x??

#### Monadic Bind Operator for Asynchronous Operations
Background context: The `async Task<T>` type is a monadic container that allows for applying the monadic operators `Bind` and `Return`. These operators help in composing asynchronous operations as a chain of computations, making the code both declarative and expressive.

The `Bind` operator works with functions that take an argument of type 'T and return a computation of type `Task<'R>` (with signature `'T -> Task<'R>`). It says: "When the value 'R from the function is evaluated, it passes the result into the function."

:p What does the `Bind` operator do in the context of asynchronous operations?
??x
The `Bind` operator composes two functions that have their results wrapped in a `Task` type. When the first asynchronous operation completes and returns a value 'R, this value is passed as input to the second function, effectively chaining the execution of asynchronous tasks.

```csharp
// Example of Bind operator usage
async Task<Tuple<string, StockData[]>> ProcessStockHistory(string symbol)
{
    return await DownloadStockHistory(symbol)
               .Bind(stockHistory => ConvertStockHistory(stockHistory))
               .Bind(stockData => Task.FromResult(Tuple.Create(symbol, stockData)));
}
```

This example shows how `Bind` is used to sequentially process asynchronous operations. The first operation (`DownloadStockHistory`) runs, and once it completes, its result is passed to the next operation (`ConvertStockHistory`), which in turn passes its result to the final `Task.FromResult`.

x??

---

#### Continuation-Passing Style with Bind
Background context: Using the `Bind` operator allows you to structure asynchronous operations using a continuation-passing style (CPS). CPS defers execution until it is needed, providing finer control over the execution and enabling compositionality.

:p How does the `Bind` operator support continuation-passing style in asynchronous programming?
??x
The `Bind` operator supports continuation-passing style by allowing you to define a sequence of functions that are executed one after another. Each function receives the result from the previous operation as input, effectively creating a chain where each step is dependent on the successful completion of the previous step.

```csharp
// Example of Bind in CPS style
async Task<Tuple<string, StockData[]>> ProcessStockHistory(string symbol)
{
    return await DownloadStockHistory(symbol)
               .Bind(stockHistory => ConvertStockHistory(stockHistory))
               .Bind(stockData => Task.FromResult(Tuple.Create(symbol, stockData)));
}
```

In this example, the `DownloadStockHistory` function returns a `Task<StockHistory>`. The first `Bind` operation (`stockHistory => ConvertStockHistory(stockHistory)`) takes the result of `DownloadStockHistory` and converts it into another `Task<StockData[]>`. Finally, the second `Bind` operation uses this result to create a `Tuple<string, StockData[]>`.

x??

---

#### Hot vs. Deferred Tasks in C# TAP
Background context: In C# Task-based Asynchronous Pattern (TAP), functions that return tasks start execution immediately, making them "hot" tasks. This behavior can negatively impact compositional forms of asynchronous programming.

The functional approach to handling asynchronous operations is to defer execution until it's needed, which provides better compositionality and finer control over the execution aspect.

:p What is the difference between a hot task and a deferred task in C# TAP?
??x
A "hot" task starts executing immediately when created. This can be problematic because it means that even if you chain asynchronous operations, they may all start running at once, which defeats the purpose of asynchrony.

In contrast, a "deferred" task delays execution until its result is actually needed, making the operations composable and easier to manage. The functional approach in TAP defers execution using `async` methods and `Task` objects that do not begin executing immediately.

```csharp
// Example of a hot task (not recommended for composition)
async Task HotTaskExample()
{
    await SomeAsyncOperation(); // Starts immediately when called
}

// Example of a deferred task (preferred for composition)
async Task DeferredTaskExample()
{
    var result = await SomeAsyncOperation(); // Only starts when awaited
}
```

In the `DeferredTaskExample`, the operation only begins executing when it is awaited, providing better control and enabling more effective composition with other asynchronous operations.

x??

---

#### Asynchronous Task Models in C#
Background context: This card explains the different models of asynchronous tasks used in programming, specifically focusing on the Hot, Cold, and Task Generator models. It highlights their differences and use cases.

:p What are the three main models for implementing APM (Asynchronous Programming Model) in C#?
??x
The three main models for implementing APM in C# are:
1. **Hot Tasks**: Asynchronous methods return a task that represents an already running job that will eventually produce a value.
2. **Cold Tasks**: Asynchronous methods return a task that requires an explicit start from the caller, often used in thread-based approaches.
3. **Task Generators**: Asynchronous methods return a task that will eventually generate a value and starts when a continuation is provided, preferred in functional paradigms to avoid side effects and mutation.

Example of each model:
```csharp
// Hot Tasks Example (C#)
public Task<int> GetResultAsync()
{
    // An already running job.
    var result = 42;
    return Task.FromResult(result);
}

// Cold Tasks Example (C#)
public async Task<int> GetValueAsync()
{
    await Task.Delay(1000); // Simulating some time-consuming work
    return 42;
}

// Task Generators Example (F#)
let downloadStockHistory = async {
    do! Async.Sleep(1000) // Simulate delay
    return "MSFT Stock History"
}
```
x??

---

#### Lazy Evaluation with Func<T> in C#
Background context: This card describes how to lazily evaluate an asynchronous operation using a `Func<Task<T>>` delegate, which only runs the underlying operation when explicitly called.

:p How can you use a `Func<Task<T>>` to lazily evaluate an asynchronous operation in C#?
??x
You can use a `Func<Task<T>>` to lazily evaluate an asynchronous operation by defining it as follows:

```csharp
Func<Task<string>> onDemand = async () => await DownloadStockHistory("MSFT");
string stockHistory = await onDemand();
```

Here, the function `onDemand` is defined as a `Func<Task<string>>`, which means it returns a task that represents an asynchronous operation. The actual execution of this operation (i.e., downloading the stock history) only occurs when you explicitly call the `onDemand()` method.

:p What is a small glitch in the provided code snippet?
??x
The function `onDemand` runs the asynchronous expression, which must have a fixed argument (`"MSFT"`). If you want to pass different stock symbols dynamically, there's a need for currying and partial application techniques.

x??

---

#### Currying and Partial Application in C#
Background context: This card explains the concepts of currying and partial application in functional programming, allowing easier reuse of more abstract functions by specializing them with specific parameters.

:p What are currying and partial application in FP (Functional Programming) languages?
??x
In functional programming languages, a function is **curried** when it takes multiple arguments but appears to take one argument at a time. Each call returns another function until all the arguments have been provided. For example, a function type signature `A -> B -> C` can be translated into C# as `Func<A, Func<B, C>>`. This allows for partial application where you can create new functions by applying some of the parameters and creating a specialized version with fewer arguments.

:p How does currying work in C#?
??x
In C#, you can define a curried function that takes one argument and returns another function. Here’s an example:

```csharp
Func<string, Func<Task<string>>> onDemandDownload = symbol => 
    async () => await DownloadStockHistoryAsync(symbol);
```

This `onDemandDownload` function takes a string (symbol) as an argument and returns a new `Func<Task<string>>`. You can then partially apply this function to create specialized functions. For instance:

```csharp
Func<Task<string>> onDemandDownloadMSFT = onDemandDownload("MSFT");
string stockHistoryMSFT = await onDemandDownloadMSFT();
```

:p How does partial application work with the `onDemandDownload` function?
??x
Partial application works by calling a curried function with some of its parameters and returning a new function that expects the remaining parameters. In the example:

```csharp
Func<string, Func<Task<string>>> onDemandDownload = symbol => 
    async () => await DownloadStockHistoryAsync(symbol);
```

You can partially apply this function to create a specialized version for "MSFT":

```csharp
Func<Task<string>> onDemandDownloadMSFT = onDemandDownload("MSFT");
string stockHistoryMSFT = await onDemandDownloadMSFT();
```

Here, `onDemandDownload("MSFT")` creates a new function that expects no more parameters and executes the asynchronous operation with "MSFT".

x??

---

#### Retry Mechanism for Asynchronous Operations
Background context explaining the concept. When working with asynchronous I/O operations, particularly network requests, unexpected issues like bad internet connections or unavailable remote servers can occur, leading to failed attempts. A common practice is to implement a retry mechanism that allows the operation to be retried a specified number of times with a delay between each attempt.
:p What is the purpose of implementing a retry mechanism in asynchronous operations?
??x
The purpose is to handle temporary failures and increase the likelihood of success by retrying the operation when an initial attempt fails. This is particularly useful for network requests where issues might be transient.

```csharp
async Task<T> Retry<T>(Func<Task<T>> task, int retries, TimeSpan delay, CancellationToken cts = default(CancellationToken))
{
    return await task().ContinueWith(async innerTask =>
    {
        cts.ThrowIfCancellationRequested();

        if (innerTask.Status == TaskStatus.Faulted)
            return innerTask.Result;

        if (retries == 0)
            throw innerTask.Exception ?? throw new Exception();

        await Task.Delay(delay, cts);
        
        return await Retry(task, retries - 1, delay, cts);
    }).Unwrap();
}
```
x??

---

#### Implementation of the Retry Function
Explanation on how to implement a retry function in C#. The `Retry` function is an extension method that takes an asynchronous operation (wrapped as a `Func<Task<T>>`), the number of retries allowed, and the delay between attempts. It handles cancellation tokens for graceful termination.
:p How does the `Retry` function work?
??x
The `Retry` function works by wrapping the async operation inside a continuation task. If the initial attempt fails (`innerTask.Status == TaskStatus.Faulted`), it retries the operation with the specified delay until either the maximum number of retries is reached or the operation succeeds.

```csharp
async Task<T> Retry<T>(Func<Task<T>> task, int retries, TimeSpan delay, CancellationToken cts = default(CancellationToken))
{
    return await task().ContinueWith(async innerTask =>
    {
        // Check for cancellation request
        cts.ThrowIfCancellationRequested();

        if (innerTask.Status == TaskStatus.Faulted)
            return innerTask.Result;

        if (retries == 0)
            throw innerTask.Exception ?? throw new Exception();

        await Task.Delay(delay, cts);

        // Recursively call Retry with reduced retries
        return await Retry(task, retries - 1, delay, cts);
    }).Unwrap();
}
```
x??

---

#### Usage of the Retry Function in DownloadStockHistory
Explanation on how to use the `Retry` function within a specific method. The `DownloadStockHistory` function is used for making HTTP requests and downloading stock history data. By wrapping it inside the `Retry` function, we can ensure that if an initial request fails due to network issues, it will be retried with a delay.
:p How does the refactored `ProcessStockHistory` method use the retry mechanism?
??x
The `ProcessStockHistory` method uses the `Retry` function to wrap the call to `DownloadStockHistory`. It retries the download up to five times with a two-second delay between each attempt. If all retries fail, it throws an exception.

```csharp
async Task<Tuple<string, StockData[]>> ProcessStockHistory(string symbol)
{
    string stockHistory = await Retry(() => DownloadStockHistory(symbol), 5, TimeSpan.FromSeconds(2));
    StockData[] stockData = await ConvertStockHistory(stockHistory);
    
    return Tuple.Create(symbol, stockData);
}
```
x??

---

#### Error Handling in Asynchronous Operations
Background context explaining that asynchronous operations, especially I/O-bound ones, are prone to errors. The previous section covered retry logic as a solution for handling failures. Another approach involves using a `CancellationToken` to stop execution and a fallback mechanism if the initial operation fails.

:p What is the purpose of the `Otherwise` combinator in error handling?
??x
The `Otherwise` combinator allows for a fallback task to be executed when the primary async task fails. This helps in gracefully recovering from errors without stopping the entire process.

```csharp
static Task<T> Otherwise<T>(this Task<T> task, Func<Task<T>> otherTask)
{
    return task.ContinueWith(async innerTask => 
    {
        if (innerTask.Status == TaskStatus.Faulted) 
            return await otherTask();
        
        return innerTask.Result;
    }).Unwrap();
}
```

This code snippet defines the `Otherwise` combinator, which takes two tasks and executes a fallback task if the first one fails.

x??

---
#### Fallback Task Execution
Background context explaining that when an async operation fails, a fallback task can be executed to handle the error gracefully. The status of a task can be checked using the `Status` property, where `TaskStatus.Faulted` indicates an exception was thrown during execution.

:p How does the `Otherwise` combinator determine whether to execute the fallback task?
??x
The `Otherwise` combinator checks the status of the primary task. If the task's status is `Faulted`, it means an exception occurred, and thus, the fallback task defined by `otherTask` will be executed.

```csharp
if (innerTask.Status == TaskStatus.Faulted)
    return await otherTask();
```

This logic ensures that if the primary async operation fails, the fallback task is triggered to handle the error.

x??

---
#### Using `CancellationToken`
Background context explaining how a `CancellationToken` can be used to stop an ongoing async operation. The default value of `CancellationToken` is `CancellationToken.None`, which means no cancellation token is provided by default.

:p What does setting the `CancellationToken` to `CancellationToken.None` imply?
??x
Setting the `CancellationToken` to `CancellationToken.None` indicates that there is no intention to cancel the operation. This means that the async operation will continue running until it completes, either successfully or due to an exception.

```csharp
CancellationToken cts = CancellationToken.None;
```

This line of code sets a default `CancellationToken`, meaning the operation will not be interrupted unless explicitly canceled by setting the token's cancellation flag.

x??

---
#### Combining Retry Logic and Fallback
Background context explaining how retry logic can be combined with fallback mechanisms to handle errors in asynchronous operations. This ensures that even if an initial async operation fails, a fallback mechanism is available for recovery.

:p How does the `Otherwise` combinator fit into the error handling strategy?
??x
The `Otherwise` combinator integrates into the error handling strategy by allowing you to specify a fallback task when the primary async operation fails. This ensures that the system can recover from errors gracefully, providing a seamless experience even if initial attempts fail.

```csharp
otherTask is wrapped into a Func<> to be evaluated only on demand.
If innerTask fails, then orTask is computed.
```

This means that `orTask` will only run when necessary—when the primary task has failed. This approach helps in managing errors without disrupting the overall flow of the application.

x??

---

