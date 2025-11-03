# Flashcards: ConcurrencyNetModern_processed (Part 6)

**Starting Chapter:** 2.3 Memoization-caching technique for program speedup

---

#### Memoization-Caching Technique for Program Speedup
Background context: Memoization is a technique used to optimize functions by caching their results. This approach is particularly useful when repeated function calls with the same arguments occur, as it avoids redundant computations and improves performance.

Explanation: In functional programming, memoization can significantly enhance the speed of programs that perform repetitive calculations. When a function is called multiple times with the same input values, rather than recalculating the result each time, the function can return the cached value from previous calls.

:p What is memoization?
??x
Memoization is a technique in functional programming where the results of expensive function calls are cached and reused when the same inputs occur again. This optimization reduces redundant computations and improves program performance.
x??

---

#### Functional Behavior of Immutable Types in F#
Background context: In functional programming languages like F#, variables are typically immutable, meaning once a value is assigned, it cannot be changed. Instead, new values are created with different memory locations.

Explanation: The use of immutability leads to the creation of new values during each iteration of loops. This behavior contrasts with mutable variables in imperative languages like C#. In F#, a loop does not update an existing variable but creates a new value instead.

:p How do F# procedural for loops differ from C#?
??x
In F#, procedural for loops create new immutable values at each iteration, whereas in C#, the same mutable variable is updated. This difference affects how closures capture and reference these variables in concurrent or multithreaded environments.
x??

---

#### Closures in Multithreading Environments
Background context: Closures are a feature of functional programming languages that allow functions to access variables from their lexical scope, even after the outer function has finished execution.

Explanation: In .NET TPL (Task Parallel Library), closures can be used to execute multiple threads by capturing variables and passing them to different tasks. This is demonstrated in Listing 2.11 where `ProcessImage` uses `Parallel.Invoke` to split an array into halves for processing.

:p How does the `Parallel.Invoke` method use closures?
??x
The `Parallel.Invoke` method in .NET TPL spawns multiple threads, each running a lambda function that captures and uses variables from the outer scope. In Listing 2.11, `ProcessImage` splits an image array into two halves and processes them concurrently using closures to share the array reference.
x??

---

#### Memory Allocation and Performance Impact of Closures
Background context: While closures offer powerful functionality for task parallelism, they come with performance costs due to increased memory allocations.

Explanation: The F# compiler creates a new object encapsulating the function and its environment when a closure is created. This process increases both memory usage and runtime overhead compared to regular functions.

:p Why are closures heavier in terms of memory and slower than regular functions?
??x
Closures are heavier on memory because they create a new object that encapsulates the function and its environment, which requires more allocations. They are also slower due to additional runtime overhead when invoking them.
x??

---

#### Example of Parallel Processing with Closures
Background context: Listing 2.11 shows how `Parallel.Invoke` can be used in .NET TPL to process parts of an array concurrently using closures.

:p What is the role of the `ProcessArray` method in Listing 2.11?
??x
The `ProcessArray` method processes a portion of the byte array, split into two halves by `Parallel.Invoke`. Each lambda closure captures and uses the shared `array` variable to process its specific part.
x??

---

#### Caching Mechanism with Memoization
Background context: Memoization stores the results of function calls for future use, avoiding redundant computations.

Explanation: A memoized function checks if a result exists in an internal table before executing. If it does, it returns the cached value; otherwise, it computes the result and caches it for later use.

:p How does memoization ensure efficient function execution?
??x
Memoization ensures efficient function execution by caching results of previous computations based on input values. When the same inputs are provided again, the stored result is returned without re-computing, reducing redundant operations.
x??

---

#### Implementation of Memoization in a Function
Background context: Figure 2.3 illustrates how memoization works with a diagram showing function evaluation and table storage.

:p How does memoization work according to Figure 2.3?
??x
Memoization works by storing the results of expensive function calls for specific inputs. When the same input is passed again, the stored result is returned immediately instead of recomputing it.
x??

---

#### Memoization Overview
Background context explaining the concept. Memoization is a technique that converts a function into a data structure to store results of expensive function calls and reuse them when the same inputs occur again. This is particularly useful for functions with repetitive computations, where storing results can significantly reduce execution time.
:p What is memoization?
??x
Memoization is a programming optimization technique used primarily to speed up computer programs by storing the results of expensive function calls and reusing those results when the same inputs occur again. This technique can be applied to any function that has no side effects, meaning its output only depends on its input parameters.
x??

---

#### Using Closures for Memoization
Background context explaining how closures are used in memoization to create a local variable (lookup table) and store function results based on the arguments passed.
:p How does memoization use closures?
??x
Memoization uses closures to encapsulate a local lookup table that stores the results of previous computations. Each call to a memoized function is wrapped by a closure, which checks if the result for the given input already exists in the cache (lookup table). If it does, the cached result is returned; otherwise, the function is evaluated and its result is stored in the cache before being returned.
x??

---

#### Implementation of Memoization
Code example showing how to implement memoization using a higher-order function (HOF) in C#.
:p How can you implement memoization in C#?
??x
Here's an implementation of memoization in C#. This HOF takes another function as input and returns a new function that caches its results.

```csharp
static Func<T, R> Memoize<T, R>(Func<T, R> func) 
    where T : IComparable
{
    Dictionary<T, R> cache = new Dictionary<T, R>();
    
    return arg => {
        if (cache.ContainsKey(arg))
            return cache[arg];
        
        var result = func(arg);
        cache.Add(arg, result);
        return result;
    };
}
```
The `Memoize` function returns a closure that wraps the original function. This closure first checks if the result for the given argument already exists in the `cache`. If it does, it returns the cached result; otherwise, it computes the result using the original function and stores it in the cache before returning.
x??

---

#### Example Application: Image Processing
Background context explaining how memoization can be applied to optimize image processing by caching results of pixel color filtering operations.
:p How can memoization help with image processing?
??x
Memoization can greatly improve the performance of image processing tasks, especially when a specific filter is repeatedly applied to pixels that have identical values. By caching the results of these computations, the system avoids redundant calculations and speeds up the overall process.

For example, if an image filter function is called multiple times with the same pixel values, memoization can cache the computed result for each unique set of input parameters (pixel coordinates), thereby reducing unnecessary computation.
x??

---

#### Cache Implementation in Code
Code example demonstrating a simple cache implementation using `Dictionary` in C#.
:p How does the `Dictionary` work as a cache in this context?
??x
The `Dictionary<T, R>` is used to store the results of function calls based on their input parameters. It acts as a lookup table where keys are the arguments passed into the memoized function and values are the corresponding computed results.

Here's how it works:
- When the memoized function is called with an argument for the first time, the `Dictionary` checks if this key already exists.
- If the key does not exist, the original function is executed, and its result is stored in the dictionary using the key as a reference.
- On subsequent calls with the same argument, the cache is checked first. If the key exists, it returns the cached value immediately; otherwise, it re-evaluates the function.

```csharp
Dictionary<T, R> cache = new Dictionary<T, R>();
```
This `Dictionary` allows constant-time lookups and storage of results.
x??

---

#### Memoization in Functional Programming
Background context: Memoization is a technique used to optimize recursive functions and functions with expensive computations by caching their results based on function inputs. This can significantly improve performance for functions that are called repeatedly with the same arguments.

The provided F# code snippet shows how memoization can be implemented using a dictionary to store previously computed results. The `memoize` function creates an internal table to cache the function results, which are then returned if the same input is encountered again.

:p What does the `memoize` function in F# do?
??x
The `memoize` function in F# takes another function as an argument and returns a new version of that function with caching functionality. It uses a dictionary to store computed results based on their inputs, ensuring that subsequent calls with the same arguments return cached results instead of recomputing them.

```fsharp
let memoize func = 
    let table = Dictionary<_,_>()
    fun x -> 
        if table.ContainsKey(x) then 
            table.[x] 
        else 
            let result = func x 
            table.[x] <- result 
            result
```
x??

---
#### Example of Greeting Function in C#
Background context: The provided example demonstrates the performance difference between a non-memoized and memoized version of a function. It shows how adding a delay (simulated by `Thread.Sleep(2000)`) affects the output when calling the function multiple times.

:p How does the Greeting function behave without memoization?
??x
Without memoization, each call to the `Greeting` function will compute and print the greeting message with the current timestamp. Since there is no caching mechanism, every call to the function will perform the computation from scratch, leading to different timestamps in subsequent calls.

```csharp
public static string Greeting(string name) { 
    return $"Warm greetings {name}, the time is  ➥ {DateTime.Now.ToString("hh:mm:ss")}";
}
Console.WriteLine(Greeting ("Richard")); // Example output: Warm greetings Richard, the time is 10:55:34
System.Threading.Thread.Sleep(2000); 
Console.WriteLine(Greeting ("Paul"));   // Output: Warm greetings Paul, the time is 10:55:36
```
x??

---
#### Memoized Greeting Function in C#
Background context: The memoized version of the `Greeting` function uses a dictionary to store and retrieve previously computed results. This allows subsequent calls with the same arguments to return cached results, reducing computation time.

:p How does the memoized version of the Greeting function behave?
??x
The memoized version of the `Greeting` function returns the cached result for any input that has been seen before. The first call computes and stores the result in the cache, but subsequent calls with the same arguments return the previously computed result from the cache, leading to identical timestamps.

```csharp
var greetingMemoize = Memoize<string, string>(Greeting);
Console.WriteLine(greetingMemoize ("Richard")); // Example output: Warm greetings Richard, the time is 10:57:21
System.Threading.Thread.Sleep(2000); 
Console.WriteLine(greetingMemoize ("Paul"));   // Output: Warm greetings Paul, the time is 10:57:23
Console.WriteLine(greetingMemoize("Richard")); // Output: Warm greetings Richard, the time is 10:57:21
```
x??

---
#### Difference Between Memoized and Non-Memoized Functions
Background context: The performance of memoized functions can vary depending on the specific scenario. In scenarios where a function is called frequently with the same arguments, memoization can significantly reduce computation time by avoiding redundant calculations.

:p What are the implications of using memoization in certain scenarios?
??x
Using memoization can be beneficial in scenarios where a function is called repeatedly with the same inputs, as it caches the results and avoids recomputation. However, for functions that do not exhibit this characteristic (e.g., those with varying inputs or side effects), memoization might introduce overhead due to additional cache management.

In the provided example, the `Greeting` function demonstrates an improvement in performance when memoized because it is called multiple times with the same name argument, leading to redundant computation. In contrast, if the `Greeting` function were to generate a unique message each time (e.g., based on a random number or other varying data), memoization would not offer significant benefits and might even introduce unnecessary overhead.

x??

---

#### Web Crawler Basics
Background context explaining how a web crawler works. The `WebCrawler` function downloads content from URLs, analyzes HTML to find links, and extracts page titles.

:p Describe the `WebCrawler` function and its purpose?
??x
The `WebCrawler` function is designed to recursively fetch and analyze the content of websites by downloading the content using `GetWebContent`, extracting hyperlinks with `AnalyzeHtmlContent`, and then processing these links. It also extracts the page title from the HTML content.

```csharp
public static IEnumerable<string> WebCrawler(string url)
{
    string content = GetWebContent(url);
    yield return content;
    foreach (string item in AnalyzeHtmlContent(content))
        yield return GetWebContent(item);
}
```
x??

---

#### Memoization for Improved Performance
Explanation of memoization and how it improves the performance of recursive or repetitive operations. The provided example shows using `Memoize` to avoid redundant downloads by caching results.

:p How does memoizing the `WebCrawler` function improve efficiency?
??x
Memoization is a technique that stores the results of expensive function calls and returns the cached result when the same inputs occur again, thus avoiding redundant work. In this case, it ensures that each URL's content is fetched only once, significantly reducing the time taken for repeated requests to the same URLs.

```csharp
static Func<string, IEnumerable<string>> WebCrawlerMemoized = Memoize<string, IEnumerable<string>>(WebCrawler);
```
x??

---

#### Parallel Execution with PLINQ
Explanation of how LINQ and PLINQ can be used to process queries in parallel for better performance. This is demonstrated by changing the query to use `AsParallel()`.

:p How does using `AsParallel()` improve the web crawler's performance?
??x
Using `AsParallel()` enables the LINQ query to execute in parallel, distributing the work across multiple threads. This significantly speeds up the process of crawling and analyzing multiple URLs simultaneously.

```csharp
var webPageTitles = from url in urls.AsParallel()
                    from pageContent in WebCrawlerMemoized(url)
                    select ExtractWebPageTitle(pageContent);
```
x??

---

#### Combining Memoization and Parallel Execution
Explanation of combining memoization with parallel execution to optimize both performance and efficiency.

:p How does the combination of `WebCrawlerMemoized` and `AsParallel()` enhance the web crawler's functionality?
??x
By combining memoization (`WebCrawlerMemoized`) with parallel execution using `AsParallel()`, we ensure that the web crawler fetches each URL only once, while also processing multiple URLs concurrently. This results in faster overall performance by minimizing redundant downloads and leveraging multi-threading.

```csharp
var webPageTitles = from url in urls.AsParallel()
                    from pageContent in WebCrawlerMemoized(url)
                    select ExtractWebPageTitle(pageContent);
```
x??

---

#### ConcurrentDictionary for Thread-Safe Memoization
Background context: The provided text discusses the use of `ConcurrentDictionary` to ensure thread-safe memoization, which is crucial when dealing with parallelism. This approach helps avoid race conditions and ensures that multiple threads do not access or modify shared data concurrently in an unsafe manner.

If applicable, add code examples with explanations:
```csharp
public Func<T, R> MemoizeThreadSafe<T, R>(Func<T, R> func) where T : IComparable
{
    ConcurrentDictionary<T, R> cache = new ConcurrentDictionary<T, R>();
    return arg => cache.GetOrAdd(arg, a => func(a));
}
```
:p What is the purpose of using `ConcurrentDictionary` in this context?
??x
The purpose of using `ConcurrentDictionary` is to ensure thread safety when multiple threads access or modify shared data. The `GetOrAdd` method guarantees that only one evaluation of the function for a given argument is added to the collection, even if multiple threads check the cache concurrently.

Code example:
```csharp
public Func<T, R> MemoizeThreadSafe<T, R>(Func<T, R> func) where T : IComparable
{
    ConcurrentDictionary<T, R> cache = new ConcurrentDictionary<T, R>();
    return arg => cache.GetOrAdd(arg, a => func(a));
}
```
x??

---
#### Lazy Evaluation for Enhanced Performance
Background context: The text introduces the concept of lazy evaluation to further optimize memoization. By using `Lazy<R>` instead of direct caching, you ensure that function initialization is deferred until necessary, reducing overhead in highly concurrent applications where expensive object initializations are involved.

:p How does lazy evaluation improve performance in parallel environments?
??x
Lazy evaluation improves performance by deferring the execution of potentially expensive operations until they are actually needed. This approach ensures that the function initializer `func(a)` is not called multiple times for the same value, thus saving computational resources and improving overall efficiency.

Code example:
```csharp
static Func<T, R> MemoizeLazyThreadSafe<T, R>(Func<T, R> func) where T : IComparable
{
    ConcurrentDictionary<T, Lazy<R>> cache = new ConcurrentDictionary<T, Lazy<R>>();
    return arg => cache.GetOrAdd(arg, a => new Lazy<R>(() => func(a))).Value;
}
```
x??

---
#### Performance Benefits of Parallel Execution with PLINQ
Background context: The text demonstrates the performance benefits of using PLINQ for parallel execution in web crawling tasks. By leveraging `AsParallel()` and concurrent memoization techniques, the processing time can be significantly reduced compared to sequential implementations.

:p How does PLINQ contribute to faster execution times?
??x
PLINQ (Parallel Language Integrated Query) contributes to faster execution times by allowing multiple threads to process elements in parallel. This is particularly useful for tasks like web crawling where IO-bound operations can be executed concurrently, leading to substantial performance improvements.

Code example:
```csharp
var urls = new List<string>();
// Populate the list of URLs

var webPageTitles = from url in urls.AsParallel()
                    from pageContent in WebCrawlerMemoizedThreadSafe(url)
                    select ExtractWebPageTitle(pageContent);
```
x??

---
#### Reducing Network I/O Operations with Concurrent Memoization
Background context: The text emphasizes that concurrent memoization not only speeds up execution but also reduces network I/O operations. By caching results efficiently and avoiding redundant function evaluations, the number of requests made to external resources like web servers is minimized.

:p How does concurrent memoization reduce network I/O?
??x
Concurrent memoization reduces network I/O by caching the results of expensive function calls (e.g., web page crawls) so that subsequent calls with the same arguments do not require re-execution. This minimizes the number of requests sent to external resources, thereby reducing network traffic and improving overall efficiency.

Code example:
```csharp
public Func<string, IEnumerable<string>> WebCrawlerMemoizedThreadSafe = 
    MemoizeThreadSafe<string, IEnumerable<string>>(WebCrawler);

var webPageTitles = from url in urls.AsParallel()
                    from pageContent in WebCrawlerMemoizedThreadSafe(url)
                    select ExtractWebPageTitle(pageContent);
```
x??

---

#### Thread Safety and Lazy Evaluation
Thread safety is crucial in multithreaded environments to prevent data corruption or inconsistent states. Without explicit thread safety, functions like `func(a)` could be executed concurrently by multiple threads, leading to undefined behavior.

:p What are the challenges of ensuring thread safety for a function in a multithreaded environment?
??x
Ensuring thread safety manually can be complex and error-prone. Primitive locks (mutexes, semaphores) are often used but introduce overhead and potential deadlocks or race conditions if not managed carefully. An alternative approach is to use constructs like `Lazy<T>` which guarantees thread safety by evaluating the function only once and providing a thread-safe access point.

```csharp
using System;
using System.Linq;
using System.Threading;

class Program {
    static void Main() {
        var lazyResult = new Lazy<int>(() => ThreadSafeFunction());
        
        // Simulate multithreaded environment
        Parallel.Invoke(
            () => Console.WriteLine(lazyResult.Value),
            () => Console.WriteLine(lazyResult.Value)
        );
    }
    
    static int ThreadSafeFunction() {
        // Thread-safe computation
        return 42;
    }
}
```
x??

---

#### Memory Leaks in Memoization
Memoization can lead to memory leaks if not managed properly. Storing all computed results in a simple dictionary without any cleanup mechanism can cause the application to consume more and more memory over time.

:p What is a common issue with using a standard dictionary for memoization?
??x
A common issue is that a dictionary storing results does not have a way to automatically remove old or unused entries. This leads to potential memory leaks as items are continuously added without being removed, even if they are no longer needed.

```csharp
using System;
using System.Collections.Generic;

class Memoizer {
    private Dictionary<int, int> memo = new Dictionary<int, int>();

    public int Compute(int n) {
        if (memo.ContainsKey(n)) return memo[n];
        
        // Simulate a costly computation
        int result = n * 2;
        memo[n] = result;
        return result;
    }
}
```
x??

---

#### Weak References for Memoization
Using `WeakReference` can help manage memory more effectively by allowing results to be garbage collected when the associated key is no longer alive. This approach avoids unnecessary memory retention and optimizes memory usage.

:p How does using `WeakReference` in memoization help with memory management?
??x
Using `WeakReference` helps manage memory more efficiently because it allows referenced objects (like computed results) to be garbage collected if the reference (key) is no longer alive. This mechanism ensures that unnecessary data is not kept in memory, optimizing resource usage.

```csharp
using System;
using System.Collections.Concurrent;

class MemoizerWithWeakReferences {
    private ConcurrentDictionary<WeakReference<int>, int> memo = new ConcurrentDictionary<WeakReference<int>, int>();

    public int Compute(int n) {
        var key = new WeakReference<int>(n);
        
        if (memo.TryGetValue(key, out var result)) return result;
        
        // Simulate a costly computation
        int computedResult = n * 2;
        memo[key] = computedResult;
        return computedResult;
    }
}
```
x??

---

#### Cache Expiration Policy
Cache expiration policies involve storing additional metadata (like timestamps) with each cached item to automatically invalidate and remove old items. This approach ensures that the cache is not indefinitely holding onto stale data.

:p What is a cache expiration policy used for in memoization?
??x
A cache expiration policy is used to automatically invalidate and remove cached items after a certain period, ensuring that only recently computed or relevant results are kept. This helps in managing memory usage by reducing the risk of memory leaks caused by storing outdated data indefinitely.

```csharp
using System;
using System.Collections.Concurrent;

class MemoizerWithExpiration {
    private ConcurrentDictionary<int, (int Result, DateTime Timestamp)> memo = new ConcurrentDictionary<int, (int Result, DateTime Timestamp)>();

    public int Compute(int n) {
        if (memo.TryGetValue(n, out var entry)) {
            TimeSpan age = DateTime.Now - entry.Timestamp;
            
            // Invalidate after 10 seconds
            if (age.TotalSeconds > 10) return entry.Result;
            
            return entry.Result + 1; // Simulate a new computation
        }
        
        int result = n * 2;
        memo[n] = (result, DateTime.Now);
        return result;
    }
}
```
x??

---

---
#### Speculative Processing and Concurrent Speculation
Background context: Speculative processing is a technique where computations are performed before they are actually needed, to amortize the cost of expensive computations. This can significantly improve performance and responsiveness in programs. The idea is to start precomputing tasks as soon as enough input data becomes available.

:p What is speculative processing?
??x
Speculative processing involves performing computations ahead of time when there is sufficient input data available, so that these computations are ready by the time they are needed.
x??

---
#### Jaro-Winkler Distance Implementation
Background context: The Jaro-Winkler distance is a measure used to determine the similarity between two strings. It is particularly suited for short strings like proper names and provides a score normalized from 0 (no similarity) to 1 (exact match).

:p What is the purpose of the Jaro-Winkler algorithm?
??x
The Jaro-Winkler algorithm measures the similarity between two strings, providing a score that ranges from 0 to 1, where 1 indicates an exact match.
x??

---
#### Fuzzy Match Function Implementation in C#
Background context: The provided `FuzzyMatch` function uses PLINQ (Parallel LINQ) to find the best fuzzy match for a given word among a list of words. It leverages the Jaro-Winkler distance algorithm.

:p What does the `FuzzyMatch` function do?
??x
The `FuzzyMatch` function finds the best fuzzy match for a given word in a list of words using the Jaro-Winkler distance algorithm. It uses PLINQ to perform computations in parallel, improving performance.
x??

---
#### Performance Issue with Fuzzy Match Function
Background context: The current implementation of the `FuzzyMatch` function has an efficiency issue because it rebuilds the word set (HashSet) every time the function is called, leading to redundant computation.

:p Why is the current `FuzzyMatch` implementation inefficient?
??x
The current `FuzzyMatch` implementation is inefficient because it recreates the word set (HashSet) each time the function is called. This leads to repeated computations and wasted resources, which can be avoided.
x??

---
#### Partial Application and Memoization for Optimization
Background context: To optimize the performance of the fuzzy match function, we can use partial application and memoization techniques. These allow us to precompute parts of the logic once and reuse them, reducing redundant computation.

:p How can you improve the efficiency of the `FuzzyMatch` function?
??x
You can improve the efficiency by using partial application to create a reusable function that retains the word set (HashSet) after its initial creation. This reduces the need for repeated computations.
x??

---
#### Fast Fuzzy Match Implementation with Precomputation
Background context: The provided code snippet shows how to implement a more efficient fuzzy match function using partial application and memoization. It creates a function that precomputes the word set and retains it in a closure, reducing redundant computation.

:p What does the `PartialFuzzyMatch` function do?
??x
The `PartialFuzzyMatch` function precomputes the word set (HashSet) from a list of words and returns a new function that performs fuzzy matching. This retained word set is used across multiple calls to reduce redundant computations.
x??

---
#### Example Usage of Fast Fuzzy Match Function
Background context: The provided code demonstrates how to use the `PartialFuzzyMatch` function to create a fast fuzzy match function that can be reused for different words.

:p How do you use the `PartialFuzzyMatch` function?
??x
You use the `PartialFuzzyMatch` function by passing in a list of words and storing the resulting function. You can then call this function multiple times with different words to get fast fuzzy match results.
```csharp
static Func<string, string> PartialFuzzyMatch(List<string> words)
{
    var wordSet = new HashSet<string>(words);
    return word => 
        (from w in wordSet.AsParallel() 
            select JaroWinklerModule.Match(w, word)) 
            .OrderByDescending(w => w.Distance) 
            .Select(w => w.Word) 
            .FirstOrDefault();
}

Func<string, string> fastFuzzyMatch = PartialFuzzyMatch(words);
string magicFuzzyMatch = fastFuzzyMatch("magic");
string lightFuzzyMatch = fastFuzzyMatch("light");
```
x??

---

