# Flashcards: ConcurrencyNetModern_processed (Part 25)

**Starting Chapter:** 10.1.1 The problem of error handling in imperative programming

---

#### Error Handling in Asynchronous Programming
Error handling is crucial in asynchronous programming to ensure robustness and maintainability of applications. In traditional imperative programming, error handling often involves try-catch blocks that can disrupt the normal program flow.

:p How does error handling differ between functional and imperative paradigms?
??x
In functional programming (FP), error handling aims to minimize side effects and avoid exceptions by returning structural representations of success or failure. This contrasts with imperative languages, which commonly use try-catch blocks and throw statements that can introduce bugs due to disrupted program flow.

Imperative Example:
```csharp
static async Task<Image> DownloadImageAsync(string blobReference)
{
    try
    {
        var container = await Helpers.GetCloudBlobContainerAsync().ConfigureAwait(false);
        CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobReference);

        using (var memStream = new MemoryStream())
        {
            await blockBlob.DownloadToStreamAsync(memStream).ConfigureAwait(false);
            return Bitmap.FromStream(memStream);
        }
    }
    catch (StorageException ex)
    {
        Log.Error("Azure Storage error", ex);
        throw;
    }
    catch (Exception ex)
    {
        Log.Error("Some general error", ex);
        throw;
    }
}

async RunDownloadImageAsync()
{
    try
    {
        var image = await DownloadImageAsync("Bugghina0001.jpg");
        ProcessImage(image);
    }
    catch (Exception ex)
    {
        HandlingError(ex);
        throw;
    }
}
```
x??

---
#### Asynchronous Computation and Try-Catch Blocks
In the .NET Framework, wrapping all code that belongs to an asynchronous computation in a try-catch block helps manage exceptions. However, this can lead to lengthy boilerplate code.

:p What is the main issue with using traditional imperative error handling for asynchronous operations?
??x
The main issue is that it introduces unnecessary complexity and boilerplate code. This disrupts the normal program flow and makes tracing errors harder due to nested try-catch blocks. For example, in a method like `DownloadImageAsync`, most of the lines are dedicated to error handling rather than the actual functionality.

Imperative Example:
```csharp
static async Task<Image> DownloadImageAsync(string blobReference)
{
    try
    {
        var container = await Helpers.GetCloudBlobContainerAsync().ConfigureAwait(false);
        CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobReference);

        using (var memStream = new MemoryStream())
        {
            await blockBlob.DownloadToStreamAsync(memStream).ConfigureAwait(false);
            return Bitmap.FromStream(memStream);
        }
    }
    catch (StorageException ex)
    {
        Log.Error("Azure Storage error", ex);
        throw;
    }
    catch (Exception ex)
    {
        Log.Error("Some general error", ex);
        throw;
    }
}
```
x??

---
#### Functional Combinators for Asynchronous Operations
Functional combinators help in building complex asynchronous functions by composing smaller and more concise operators. This approach makes the code more maintainable and performant.

:p What are functional combinators, and how do they improve error handling?
??x
Functional combinators are utility functions that allow you to create complex functions by composing smaller and more concise operators. They help in managing side effects and errors without disrupting the normal program flow. For example, instead of nested try-catch blocks, you can use combinators like `Result<T>` or `AsyncFunc` which encapsulate error handling within their structure.

Functional Example:
```csharp
static async Task<Image> DownloadImageAsync(string blobReference)
{
    var container = await Helpers.GetCloudBlobContainerAsync().ConfigureAwait(false);
    CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobReference);

    using (var memStream = new MemoryStream())
    {
        await blockBlob.DownloadToStreamAsync(memStream).ConfigureAwait(false);
        return Bitmap.FromStream(memStream);
    }
}
```
x??

---
#### Exception Handling in Asynchronous Methods
Exception handling is essential for asynchronous methods to prevent runtime failures and ensure that the application can recover gracefully from errors.

:p What are some common issues faced when implementing error handling in asynchronous methods?
??x
Common issues include:
1. **Complexity**: Nested try-catch blocks can make code harder to read and maintain.
2. **Performance Overhead**: Error handling can introduce performance overhead due to the additional logic.
3. **Scalability**: Asynchronous operations can be complex, making error handling a significant challenge.

Example of Complex Code:
```csharp
try
{
    var image = await DownloadImageAsync("Bugghina0001.jpg");
    ProcessImage(image);
}
catch (Exception ex)
{
    HandlingError(ex);
    throw;
}
```
x??

---
#### Using Built-In Asynchronous Combinators
Built-in combinators like `Result<T>` or `AsyncFunc` help in managing asynchronous operations more efficiently by encapsulating error handling.

:p How can built-in asynchronous combinators be used to simplify error handling?
??x
Built-in combinators like `Result<T>` or `AsyncFunc` provide a structured way to handle asynchronous operations. They encapsulate the error handling logic, making the code cleaner and easier to maintain. For example:

```csharp
static async Task<Result<Image>> DownloadImageAsync(string blobReference)
{
    try
    {
        var container = await Helpers.GetCloudBlobContainerAsync().ConfigureAwait(false);
        CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobReference);

        using (var memStream = new MemoryStream())
        {
            await blockBlob.DownloadToStreamAsync(memStream).ConfigureAwait(false);
            return Result.Ok(Bitmap.FromStream(memStream));
        }
    }
    catch (StorageException ex)
    {
        return Result.Fail<Bitmap>(ex.Message);
    }
    catch (Exception ex)
    {
        return Result.Fail(ex);
    }
}
```
x??

---
#### Custom Asynchronous Combinators
Custom asynchronous combinators can be implemented to meet specific application requirements, improving performance and maintainability.

:p How can custom asynchronous combinators be designed?
??x
Custom asynchronous combinators can be designed by creating utility functions that encapsulate common operations. For example, you might create a combinator that handles retries or parallel execution of tasks. These combinators reduce boilerplate code and make the application more maintainable.

Example:
```csharp
public static async Task<T> RetryAsync(Func<Task<T>> operation, int maxRetries)
{
    for (int i = 0; i < maxRetries; i++)
    {
        try
        {
            return await operation();
        }
        catch (Exception ex)
        {
            if (i == maxRetries - 1) throw;
            Console.WriteLine($"Retry attempt {i + 1}: {ex.Message}");
        }
    }
}
```
x??

---
#### Interoperability Between C# and F#
Interoperability allows developers to call and pass asynchronous functions between different programming languages, such as C# and F#, enhancing flexibility.

:p What is the importance of interoperability in functional combinators for concurrent programming?
??x
The importance lies in leveraging the strengths of both languages. For instance, you might use C# for its rich ecosystem and libraries, while utilizing F#'s powerful pattern matching and functional constructs. Interoperability ensures that developers can take advantage of these features without sacrificing the benefits of a unified approach.

Example:
```csharp
async Task CallFSharpFunction()
{
    var result = await MyFSharpAsyncFunction();
    Console.WriteLine(result);
}
```
x??

---

#### Error Handling Techniques in C#
Background context explaining the importance of proper error handling, especially in asynchronous operations. The `try-catch` block is used to handle exceptions but can sometimes complicate code readability and maintenance. Functions like `Retry`, `Otherwise`, and `Task.Catch` are introduced as more structured ways to manage errors.
:p What are the advantages of using `Retry` and `Otherwise` over traditional `try-catch` blocks?
??x
The advantages include better encapsulation, reduced complexity in code, and improved reusability. These functions help in handling specific error cases in a more controlled manner, making it easier to manage retries and fallbacks.
```csharp
static async Task<T> Otherwise<T>(this Task<T> task, Func<Task<T>> orTask) =>
    task.ContinueWith(async innerTask => {
        if (innerTask.Status == TaskStatus.Faulted)
            return await orTask();
        return await Task.FromResult<T>(innerTask.Result);
    }).Unwrap();

static async Task<T> Retry<T>(Func<Task<T>> task, int retries, TimeSpan delay, CancellationToken cts = default(CancellationToken))
    => await task().ContinueWith(async innerTask => {
        cts.ThrowIfCancellationRequested();
        if (innerTask.Status == TaskStatus.Faulted)
            return innerTask.Result;
        if (retries == 0)
            throw innerTask.Exception ?? throw new Exception();
        await Task.Delay(delay, cts);
        return await Retry(task, retries - 1, delay, cts);
    }).Unwrap();
```
x??

---
#### `Task.Catch` Function
Background context explaining the need for handling specific types of exceptions in asynchronous operations. The `Task.Catch` function allows specifying how to handle certain exception types.
:p What is the purpose of the `Task.Catch` function?
??x
The `Task.Catch` function provides a way to catch and handle specific types of exceptions generated during asynchronous operations, making it easier to manage error cases in a more structured manner. It helps in providing custom recovery logic for known exception types.
```csharp
static Task<T> Catch<T, TError>(this Task<T> task, Func<TError, T> onError) where TError : Exception {
    var tcs = new TaskCompletionSource<T>();
    task.ContinueWith(innerTask => {
        if (innerTask.IsFaulted && innerTask?.Exception?.InnerException is TError)
            tcs.SetResult(onError((TError)innerTask.Exception.InnerException));
        else if (innerTask.IsCanceled)
            tcs.SetCanceled();
        else if (innerTask.IsFaulted)
            tcs.SetException(innerTask?.Exception?.InnerException ?? throw new InvalidOperationException());
        else
            tcs.SetResult(innerTask.Result);
    });
    return tcs.Task;
}
```
x??

---
#### Example of Using `Retry` and `Otherwise`
Background context explaining how to use the `Retry` and `Otherwise` functions in practice, along with an example. This demonstrates handling a specific error by retrying or falling back to another operation.
:p How can you rewrite the call to `DownloadImageAsync` using `Retry` and `Otherwise`?
??x
You can rewrite the call to `DownloadImageAsync` as follows:
```csharp
Image image = await AsyncEx.Retry(async () => 
    await DownloadImageAsync("Bugghina001.jpg")
        .Otherwise(async () => 
            await DownloadImageAsync("Bugghina002.jpg")),
    5, TimeSpan.FromSeconds(2));
```
This code retries the `DownloadImageAsync` operation up to five times with a delay of two seconds between each attempt. If it fails, it falls back to downloading an alternative image.
x??

---
#### Example of Using `Task.Catch`
Background context explaining how to use the `Task.Catch` function to handle specific exceptions in asynchronous operations, along with an example. This demonstrates handling a known exception type like `StorageException`.
:p How can you use `Task.Catch` to handle `StorageException`?
??x
You can define and use the `CatchStorageException` extension method as follows:
```csharp
static Task<Image> CatchStorageException(this Task<Image> task) => 
    task.Catch<Image, StorageException>(ex => Log($"Azure Blob Storage Error {ex.Message}"));

// Example usage:
Image image = await DownloadImageAsync("Bugghina001.jpg")
    .CatchStorageException();
```
This method catches `StorageException` and logs the error message.
x??

---

#### Option Type for Error Handling in Functional Programming
Background context explaining how traditional exception handling can obfuscate programmer intentions and lead to complex, less maintainable code. The introduction of functional programming (FP) paradigms offers a cleaner approach by explicitly returning values indicating success or failure rather than using exceptions.

This is particularly useful when working with asynchronous operations like downloading images from cloud storage. Traditional methods might throw exceptions on errors, making the error handling implicit and harder to follow. In contrast, using an `Option` type makes it explicit whether an operation succeeded or failed, leading to more predictable and composable code.
:p How does the Option type help in functional programming for error handling?
??x
The Option type helps by explicitly returning either a Some value representing success or a None value indicating failure. This makes it mandatory for function callers to handle both outcomes, ensuring that errors are not hidden but rather confronted directly.

In this context, `DownloadOptionImage` returns an `async Task<Option<Image>>`, where `Option<Image>` is an `Some(Image)` if the download succeeds, and `None` otherwise.
```csharp
public async Task<Option<Image>> DownloadOptionImage(string blobReference)
{
    try
    {
        var container = await Helpers.GetCloudBlobContainerAsync().ConfigureAwait(false);
        CloudBlockBlob blockBlob = container.GetBlockBlobReference(blobReference);
        using (var memStream = new MemoryStream())
        {
            await blockBlob.DownloadToStreamAsync(memStream).ConfigureAwait(false);
            return Option.Some(Bitmap.FromStream(memStream));
        }
    }
    catch (Exception)
    {
        return Option.None;
    }
}
```
x??

---
#### Retry Mechanism with Error Handling
Background context explaining the importance of retrying operations that might fail for various reasons, especially in distributed systems where network issues or transient errors are common. The `Retry` combinator can be used to automatically retry an operation a specified number of times before giving up.

In C#, this could look like combining the `Task.Catch` method with custom logic to handle retries.
:p How does the Retry combinator work in functional programming for error handling?
??x
The Retry combinator allows you to specify how many times an asynchronous operation should be retried if it fails. It wraps the original task and retries it a certain number of times before failing permanently.

For example, in C#, you could implement a retry mechanism using `Task.Retry` or manually catching exceptions and retrying with appropriate delays between attempts.
```csharp
public async Task<SomeResult> RetryOperation(Func<Task<SomeResult>> operation, int maxRetries)
{
    for (int attempt = 0; attempt < maxRetries; attempt++)
    {
        try
        {
            return await operation();
        }
        catch (Exception ex) when (ex is TransientError || ex is NetworkTimeout)
        {
            // Log the error and wait before retrying
            await Task.Delay(TimeSpan.FromSeconds(Math.Pow(2, attempt)));  // Exponential backoff
        }
    }

    throw new InvalidOperationException("Operation failed after all retries.");
}
```
x??

---
#### Otherwise Combinator for Error Handling
Background context explaining how the `Otherwise` combinator allows you to provide a fallback value or action when an operation fails. This is useful in scenarios where it’s not acceptable for the program to completely fail but instead needs to continue with alternative logic.

In functional programming, the `Otherwise` combinator can be used with `Option` types to return a default value if the primary operation does not succeed.
:p How does the Otherwise combinator work in error handling?
??x
The `Otherwise` combinator works by providing an alternative result or action when the primary operation fails. In functional programming, it’s often used with `Option` types where you can specify what to do if there is no value (i.e., None).

For example, in C#, using `Option` and `Otherwise`, you might write:
```csharp
public Option<Image> DownloadImageOrDefault(string blobReference)
{
    var result = DownloadOptionImage(blobReference);
    
    return result.Otherwise(() => DefaultImage());
}

private Image DefaultImage()
{
    // Return a default image or create one
    return new Bitmap("default.png");
}
```
x??

---
#### Task.Catch for Error Handling in Functional Programming
Background context explaining the use of `Task.Catch` to handle exceptions within asynchronous tasks, making it easier to manage errors without breaking the flow. This is particularly useful when working with async operations that might throw unexpected exceptions.

In functional programming terms, `Task.Catch` can be seen as a way to wrap an operation and catch any thrown exceptions, converting them into values or actions you specify.
:p How does Task.Catch work in error handling?
??x
`Task.Catch` is used to handle exceptions within asynchronous tasks by wrapping the task and catching any exceptions that are thrown. It allows you to define what should happen when an exception occurs, such as logging the error or returning a default value.

For example:
```csharp
public async Task<Image> SafeDownloadImage(string blobReference)
{
    try
    {
        return await DownloadOptionImage(blobReference);
    }
    catch (Exception ex)
    {
        // Log the exception and return a default image
        Console.WriteLine($"Error downloading image: {ex.Message}");
        return new Bitmap("default.png");
    }
}
```
x??

---

