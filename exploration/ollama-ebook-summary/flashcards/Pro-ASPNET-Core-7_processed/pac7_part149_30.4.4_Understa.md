# Flashcards: Pro-ASPNET-Core-7_processed (Part 149)

**Starting Chapter:** 30.4.4 Understanding page filters

---

#### Custom Page Filters in ASP.NET Core
Background context: This concept explains how to create and use custom page filters for Razor Pages in an ASP.NET Core application. These filters can modify behavior before or after a handler method is executed, similar to action filters but tailored for Razor Pages.

:p What are the different types of page filters in ASP.NET Core, and what methods do they implement?
??x
There are two types of page filters: synchronous and asynchronous.
- Synchronous page filters implement `IPageFilter` interface, which includes three methods:
  - `OnPageHandlerSelected(PageHandlerSelectedContext context)`
  - `OnPageHandlerExecuting(PageHandlerExecutingContext context)`
  - `OnPageHandlerExecuted(PageHandlerExecutedContext context)`

- Asynchronous page filters implement `IAsyncPageFilter`, providing two asynchronous methods:
  - `OnPageHandlerSelectionAsync(PageHandlerSelectedContext context)`
  - `OnPageHandlerExecutionAsync(PageHandlerExecutingContext context, PageHandlerExecutionDelegate next)`

These methods allow for modification of the request flow and response creation in a Razor Page.
x??

---
#### Synchronous Page Filters
Background context: This type of page filter is used to modify behavior before or after a handler method executes. The `IPageFilter` interface defines three key methods that can be implemented.

:p How does the `OnPageHandlerExecuting` method work, and what context is passed to it?
??x
The `OnPageHandlerExecuting` method is called after model binding has completed but before the page handler method is invoked. It receives a `PageHandlerExecutingContext` object which contains information about the current execution state.

Example code to modify an argument:
```csharp
public void OnPageHandlerExecuting(PageHandlerExecutingContext context)
{
    if (context.HandlerArguments.ContainsKey("message1"))
    {
        context.HandlerArguments["message1"] = "New message";
    }
}
```
This example demonstrates how you can alter a handler argument before it reaches the handler method.
x??

---
#### Asynchronous Page Filters
Background context: This type of page filter uses asynchronous methods to modify behavior. The `IAsyncPageFilter` interface provides two methods for handling the request and response asynchronously.

:p What is the purpose of the `OnPageHandlerExecutionAsync` method in an async page filter?
??x
The `OnPageHandlerExecutionAsync` method allows the page filter to execute asynchronously before the handler method is invoked. It takes a `PageHandlerExecutingContext` object, which provides context about the current execution state, and a delegate (`next`) that can be used to pass control back to the next filter or directly to the handler method.

Example of how it might be implemented:
```csharp
public async Task OnPageHandlerExecutionAsync(
    PageHandlerExecutingContext context,
    PageHandlerExecutionDelegate next)
{
    // Custom logic before invoking the handler
    await next();  // Invoke the handler
    // Custom logic after invoking the handler
}
```
This method allows for fine-grained control over the request lifecycle, enabling complex asynchronous operations.
x??

---
#### Applying a Custom Page Filter
Background context: This example demonstrates creating and applying a custom page filter to modify argument values before they reach the handler method.

:p How does the `ChangePageArgs` class implement the `IPageFilter` interface?
??x
The `ChangePageArgs` class implements the `IPageFilter` interface by providing implementations for three methods:

```csharp
public class ChangePageArgs : Attribute, IPageFilter
{
    public void OnPageHandlerSelected(PageHandlerSelectedContext context)
    {
        // Do nothing here
    }

    public void OnPageHandlerExecuting(PageHandlerExecutingContext context)
    {
        if (context.HandlerArguments.ContainsKey("message1"))
        {
            context.HandlerArguments["message1"] = "New message";
        }
    }

    public void OnPageHandlerExecuted(PageHandlerExecutedContext context)
    {
        // Do nothing here
    }
}
```
This class modifies the `message1` argument to a new value before it is passed to the handler method.
x??

---
#### Using a Page Filter with Razor Pages
Background context: This example shows how to apply a page filter to modify behavior in a Razor Page.

:p How does applying the `ChangePageArgs` filter affect the output of the `MessageModel.OnGet` method?
??x
Applying the `ChangePageArgs` filter modifies the `message1` argument from "hello" to "New message" before it is passed to the handler method. This means that the `OnGet` method will receive a different value for `message1`, which in turn changes the output of the page.

Example:
```csharp
public void OnGet(string message1, string message2)
{
    Message = $"{message1}, {message2}";
}
```
With the filter applied, `message1` becomes "New message", resulting in the output: `"New message, world"`.
x??

---
#### Differentiating Page Filters and Action Filters
Background context: While both page filters and action filters can modify behavior before or after a method is executed, they are used for different scenarios. Page filters are specific to Razor Pages.

:p How do action filters differ from page filters in an ASP.NET Core application?
??x
Action filters are used for MVC controllers, whereas page filters are designed specifically for Razor Pages. They share similar methods but operate on different lifecycle stages and contexts:

- Action Filters:
  - Implemented in `ControllerBase` or its derivatives.
  - Use the `IActionFilter`, `IAuthorizationFilter`, etc., interfaces.
  - Operate on action results, model binding, and other aspects of MVC controllers.

- Page Filters:
  - Implemented in Razor Pages using the `PageModel`.
  - Use the `IPageFilter` or `IAsyncPageFilter` interfaces.
  - Directly modify the request handling logic for specific handler methods or all handlers within a page model class.

While both can be used to modify behavior, they are tailored to different parts of the application architecture and have distinct lifecycles.
x??

---

---
#### Custom Filters in Razor Pages
Razor Pages in ASP.NET Core allow you to implement custom filters that modify request and response handling. In this context, we use a PageModel class to handle GET requests and dynamically generate content based on input parameters.

:p How does the `MessageModel` class handle incoming messages in the GET request?
??x
The `MessageModel` class handles incoming messages by setting the `Message` property to a concatenated string of `message1` and `message2`. If these parameters are provided, it overrides the default message with values from the URL query.

```csharp
public void OnGet(string message1, string message2)
{
    Message = $"{message1}, {message2}";
}
```
x??

---
#### Conditional Table Rendering in Razor Pages
In this example, a table is dynamically generated based on whether `Model.Message` is of type `string` or an `IDictionary<string, string>`. If it's a dictionary, each key-value pair is displayed as a row in the table.

:p How does the Razor Page determine if `Model.Message` is an `IDictionary<string, string>` and render it accordingly?
??x
The Razor Page checks if `Model.Message` can be cast to `IDictionary<string, string>`. If so, it generates a table with each key-value pair from the dictionary.

```csharp
@if (Model.Message is IDictionary<string, string>)
{
    var dict = Model.Message as IDictionary<string, string>;
    <table class="table table-sm table-striped table-bordered">
        <thead><tr><th>Name</th><th>Value</th></tr></thead>
        <tbody>
            @if (dict != null)
            {
                foreach (var kvp in dict)
                {
                    <tr><td>@kvp.Key</td><td>@kvp.Value</td></tr>
                }
            }
        </tbody>
    </table>
}
```
x??

---
#### Result Filters in ASP.NET Core
Result filters provide a mechanism to modify the action results before they are sent to the client. These filters can inspect and alter the result, including short-circuiting the pipeline if necessary.

:p What is the purpose of `IResultFilter` in ASP.NET Core?
??x
The `IResultFilter` interface allows developers to write custom filters that execute at specific points during the request lifecycle. It provides methods to modify the action results before and after they are sent to the client.

```csharp
namespace Microsoft.AspNetCore.Mvc.Filters {
    public interface IResultFilter : IFilterMetadata {
        void OnResultExecuting(ResultExecutingContext context);
        void OnResultExecuted(ResultExecutedContext context);
    }
}
```
x??

---
#### Execution Context in `OnResultExecuting` Method
The `ResultExecutingContext` class contains properties that can be used to inspect and modify the action result. This method is called after the endpoint has produced an action result but before it is sent to the client.

:p What are some key properties of the `ResultExecutingContext` class?
??x
Key properties of the `ResultExecutingContext` class include:

- **Controller**: The object that contains the endpoint.
- **Cancel**: Setting this property to true will short-circuit the result filter pipeline.
- **Result**: The action result produced by the endpoint.

```csharp
void OnResultExecuting(ResultExecutingContext context);
```
x??

---
#### Execution Context in `OnResultExecuted` Method
The `ResultExecutedContext` class is used similarly but provides additional information about the result that has been executed. This method is called after the action result has generated a response for the client.

:p What are some key properties of the `ResultExecutedContext` class?
??x
Key properties of the `ResultExecutedContext` class include:

- **Canceled**: Returns true if another filter short-circuited the filter pipeline.
- **Controller**: The object that contains the endpoint.

```csharp
void OnResultExecuted(ResultExecutedContext context);
```
x??

---

#### Exception Property
Background context: The `Exception` property is part of a filter mechanism that handles exceptions thrown by page handlers. It provides information about any exception and can be accessed during the filter execution.

:p What does the `Exception` property return?
??x
The `Exception` property returns an instance of the exception if one was thrown by the page handler method.
x??

---

#### ExceptionHandled Property
Background context: The `ExceptionHandled` property is used to determine whether an exception that occurred during a page handler's execution has been handled by a filter. It helps in managing how exceptions are managed within the application.

:p What does the `ExceptionHandled` property indicate?
??x
The `ExceptionHandled` property is set to true if an exception thrown by the page handler has been handled by the filter, indicating that any subsequent filters or actions should not consider this as a critical error.
x??

---

#### Result Property
Background context: The `Result` property provides access to the action result used to generate the client response. It is read-only and can be modified within certain types of filters.

:p What does the `Result` property return?
??x
The `Result` property returns the action result that will be used to create a response for the client.
x??

---

#### IAsyncResultFilter Interface
Background context: The `IAsyncResultFilter` interface is part of ASP.NET Core and allows developers to implement asynchronous filters. It extends the `IFilterMetadata` interface and includes an `OnResultExecutionAsync` method.

:p What is the purpose of the `IAsyncResultFilter` interface?
??x
The `IAsyncResultFilter` interface allows developers to implement asynchronous filters that can execute before or after a result execution, providing more control over the response generation process.
x??

---

#### Always-Run Result Filters
Background context: Certain scenarios require filters to always run, even if other filters short-circuit the pipeline. The `IAlwaysRunResultFilter` and `IAsyncAlwaysRunResultFilter` interfaces are used for this purpose.

:p What is an always-run result filter?
??x
An always-run result filter is a filter that implements either the `IAlwaysRunResultFilter` or `IAsyncAlwaysRunResultFilter` interface, ensuring it runs regardless of whether other filters in the pipeline have already handled the request.
x??

---

#### Creating a Result Filter - `ResultDiagnosticsAttribute`
Background context: This example demonstrates how to create a custom result filter that examines certain query parameters and modifies the response based on those values.

:p How does the `ResultDiagnosticsAttribute` class modify the response?
??x
The `ResultDiagnosticsAttribute` class checks if the request contains a query string parameter named "diag". If present, it gathers diagnostic data about the result type and model information, then reassigns the result to a new ViewResult that displays this data.
x??

---

#### Implementation of `ResultDiagnosticsAttribute`
Background context: The code snippet provided defines how to implement the custom filter by examining query parameters and modifying the response accordingly.

:p What does the `OnResultExecutionAsync` method do in the `ResultDiagnosticsAttribute` class?
??x
The `OnResultExecutionAsync` method checks if the request contains a "diag" parameter. If present, it gathers diagnostic data about the result type and model information, then reassigns the result to a new ViewResult that displays this data.
```csharp
public async Task OnResultExecutionAsync(
    ResultExecutingContext context,
    ResultExecutionDelegate next)
{
    if (context.HttpContext.Request.Query.ContainsKey("diag"))
    {
        Dictionary<string, string?> diagData = 
            new Dictionary<string, string?>
            {
                { "Result type", context.Result.GetType().Name }
            };
        
        if (context.Result is ViewResult vr)
        {
            diagData["View Name"] = vr.ViewName;
            diagData["Model Type"] = vr.ViewData?.Model?.GetType().Name;
            diagData["Model Data"] = vr.ViewData?.Model?.ToString();
        }
        else if (context.Result is PageResult pr)
        {
            diagData["Model Type"] = pr.Model.GetType().Name;
            diagData["Model Data"] = pr.ViewData?.Model?.ToString();
        }

        context.Result = new ViewResult()
        {
            ViewName = "/Views/Shared/Message.cshtml",
            ViewData = new ViewDataDictionary(
                new EmptyModelMetadataProvider(),
                new ModelStateDictionary())
            {
                Model = diagData
            }
        };
    }

    await next();
}
```
x??

---

---
#### Result Filters and Diagnostic Information
Background context: This section explains how to use result filters in ASP.NET Core applications, particularly focusing on creating a filter that generates diagnostic information. The example provided uses a `ResultDiagnostics` attribute applied to an `HomeController`, which changes the output when a query string parameter is present.

:p What is the purpose of applying the `ResultDiagnostics` attribute to the actions in the `HomeController`?
??x
The purpose of applying the `ResultDiagnostics` attribute is to generate diagnostic information about the result being returned by the action methods. When the query string parameter "diag" is included, it triggers the filter to display detailed information such as the type of result and its properties.

```csharp
[HttpsOnly]
[ResultDiagnostics]
public class HomeController : Controller {
    // actions defined here
}
```
x??

---
#### Customizing Action Results with a Filter
Background context: The example demonstrates how to customize the output of an action method by using a result filter. Specifically, it shows how to modify the message passed to a view based on certain conditions.

:p How does the `OnActionExecuting` method in the `HomeController` change the message parameter?
??x
The `OnActionExecuting` method checks if the "message1" argument is present in the action arguments. If it is, the method replaces its value with "New message." This allows for dynamic changes to the message passed to the view based on runtime conditions.

```csharp
public override void OnActionExecuting(ActionExecutingContext context) {
    if (context.ActionArguments.ContainsKey("message1")) {
        context.ActionArguments["message1"] = "New message";
    }
}
```
x??

---
#### Applying Filters to Actions in HomeController
Background context: The example includes a method for applying filters directly to actions within the `HomeController`. This is done by decorating the action methods with the `[ResultDiagnostics]` attribute, which activates the diagnostic filter.

:p How are the action methods in `HomeController` decorated to apply the result diagnostics?
??x
The action methods in `HomeController` are decorated with the `[ResultDiagnostics]` attribute. This attribute acts as a filter that will execute its logic whenever these actions are called, allowing for the generation of diagnostic information based on certain conditions.

```csharp
[HttpsOnly]
[ResultDiagnostics]
public class HomeController : Controller {
    // actions defined here
}
```
x??

---
#### Using Result Filters with Query Parameters
Background context: The example illustrates how to detect and respond to a query string parameter "diag" within the `OnResultExecutionAsync` method. When this parameter is present, it triggers the filter to generate diagnostic information about the result being returned.

:p How does the `OnResultExecutionAsync` method handle the presence of the "diag" query parameter?
??x
The `OnResultExecutionAsync` method checks if the request contains a query parameter named "diag." If this parameter is present, it creates a dictionary to store diagnostic data. It then inspects the result type and extracts additional information such as the view name and model type.

```csharp
public override async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next) {
    if (context.HttpContext.Request.Query.ContainsKey("diag")) {
        Dictionary<string, string?> diagData = new Dictionary<string, string?> {
            { "Result type", context.Result.GetType().Name }
        };
        
        if (context.Result is ViewResult vr) {
            diagData["View Name"] = vr.ViewName;
            diagData["Model Type"] = vr.ViewData?.Model?.GetType().Name;
        }
    }
}
```
x??

---
#### Creating a Result Filter Attribute
Background context: The example shows how to create a custom result filter by deriving from the `ResultFilterAttribute` class and implementing the necessary interfaces. This attribute can be used to execute code before or after a result is executed.

:p How does one define a custom result filter in ASP.NET Core?
??x
To define a custom result filter, you derive a new attribute from `ResultFilterAttribute` and implement either the `IResultFilter`, `IActionResultFilter`, or both interfaces. In this example, the `ResultDiagnosticsAttribute` class is defined to handle diagnostic information by inspecting the result type and extracting view name and model type details.

```csharp
public class ResultDiagnosticsAttribute : ResultFilterAttribute {
    public override async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next) {
        if (context.HttpContext.Request.Query.ContainsKey("diag")) {
            Dictionary<string, string?> diagData = new Dictionary<string, string?> {
                { "Result type", context.Result.GetType().Name }
            };
            
            if (context.Result is ViewResult vr) {
                diagData["View Name"] = vr.ViewName;
                diagData["Model Type"] = vr.ViewData?.Model?.GetType().Name;
            }
        }
    }
}
```
x??

---

#### Understanding Exception Filters
Background context: In ASP.NET Core, exception filters are a mechanism to handle unhandled exceptions globally without cluttering every action method with try... catch blocks. This allows for more maintainable and cleaner code. The `IExceptionFilter` interface is used to define these global handlers.

:p What is an IExceptionFilter in ASP.NET Core?
??x
The `IExceptionFilter` interface in ASP.NET Core defines a method called `OnException` which is invoked when an unhandled exception occurs. This allows developers to handle exceptions at the filter level rather than within each action.
```csharp
public interface IExceptionFilter : IFilterMetadata {
    void OnException(ExceptionContext context);
}
```
x??

---
#### Creating Custom Exception Filters
Background context: To create custom filters, you can implement one of the exception filter interfaces or derive from `ExceptionFilterAttribute`. The most common use is to display a user-friendly error message when a specific type of exception occurs.

:p How do you create an exception filter in ASP.NET Core?
??x
You can create an exception filter by implementing the `IExceptionFilter` interface. Here's an example class that implements this interface:

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

public class RangeExceptionAttribute : ExceptionFilterAttribute {
    public override void OnException(ExceptionContext context) {
        if (context.Exception is ArgumentOutOfRangeException) {
            context.Result = new ViewResult() {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary()) {
                        Model = "The data received by the application cannot be processed"
                }
            };
        }
    }
}
```

This filter checks if the exception is an `ArgumentOutOfRangeException` and displays a custom message view.
x??

---
#### Applying Exception Filters
Background context: To apply an exception filter to a controller or action method, you simply decorate it with the `[RangeException]` attribute. This ensures that the custom logic defined in the exception filter runs when the specified exception occurs.

:p How do you apply an exception filter to an action method?
??x
To apply an exception filter, you can use the `[RangeException]` attribute on your action method. Here’s how it looks:

```csharp
[RangeException]
public ViewResult GenerateException(int? id) {
    if (id == null) { 
        throw new ArgumentNullException(nameof(id)); 
    } else if (id > 10) { 
        throw new ArgumentOutOfRangeException(nameof(id)); 
    } else { 
        return View("Message", $"The value is {id}"); 
    }
}
```

In this example, the `GenerateException` method will use the custom exception filter defined earlier. If an argument out of range or null argument occurs, it will display a specific message view.
x??

---
#### Understanding FilterContext and ExceptionContext
Background context: Both `FilterContext` and `ExceptionContext` are part of the ASP.NET Core framework that provide information about the current request and response flow. `ExceptionContext` is specifically used to handle exceptions.

:p What is an ExceptionContext in ASP.NET Core?
??x
An `ExceptionContext` in ASP.NET Core contains properties such as `Exception`, which holds the exception object, and `Result`, which allows setting a result for the response. It inherits from `FilterContext` and provides additional details about unhandled exceptions.

Example:
```csharp
public void OnException(ExceptionContext context) {
    if (context.Exception is ArgumentOutOfRangeException) {
        context.Result = new ViewResult() { 
            ViewName = "/Views/Shared/Message.cshtml", 
            ViewData = new ViewDataDictionary(
                new EmptyModelMetadataProvider(), 
                new ModelStateDictionary()) { 
                    Model = "The data received by the application cannot be processed" 
                } 
        };
    }
}
```

This code checks if the exception is an `ArgumentOutOfRangeException` and sets a custom view result.
x??

---

