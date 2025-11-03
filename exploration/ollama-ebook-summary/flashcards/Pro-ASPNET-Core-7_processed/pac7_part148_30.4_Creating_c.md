# Flashcards: Pro-ASPNET-Core-7_processed (Part 148)

**Starting Chapter:** 30.4 Creating custom filters. 30.4.1 Understanding authorization filters

---

#### Creating Custom Filters in ASP.NET Core
Background context: This concept involves implementing custom filters using the `IFilterMetadata` interface and understanding how to use different types of filters like authorization filters. The core idea is to provide additional functionality such as logging, validation, or security checks for actions, controllers, or Razor Pages.

:p What is the role of the `IFilterMetadata` interface in ASP.NET Core?
??x
The `IFilterMetadata` interface serves as a base interface that allows developers to implement custom filters. While it doesn't require specific behavior implementations, it enables the injection and execution of various types of filters.
x??

---

#### Authorization Filters
Background context: Authorization filters are used for security purposes. They run before other filters and handle request authorization. Different interfaces like `IAuthorizationFilter` and `IAsyncAuthorizationFilter` provide methods to check if a request should be allowed based on application policies.

:p How do you implement an authorization filter in ASP.NET Core?
??x
To implement an authorization filter, you need to create a class that implements the `IAuthorizationFilter` interface. Here is an example of implementing an HTTPs-only authorization filter:
```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace WebApp.Filters {
    public class HttpsOnlyAttribute : Attribute, IAuthorizationFilter {
        public void OnAuthorization(AuthorizationFilterContext context) {
            if (!context.HttpContext.Request.IsHttps) {
                // If the request is not HTTPS, return a 403 Forbidden status
                context.Result = new StatusCodeResult(StatusCodes.Status403Forbidden);
            }
        }
    }
}
```
x??

---

#### Applying Custom Filters to Controllers or Actions
Background context: Once an authorization filter is implemented, it can be applied to controllers or actions to enforce security policies. This ensures that only secure requests (e.g., HTTPS) are processed.

:p How do you apply the `HttpsOnly` filter in a controller?
??x
You can apply the `HttpsOnly` filter by decorating the desired action method with the attribute:
```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Filters;

namespace WebApp.Controllers {
    [HttpsOnly]
    public class HomeController : Controller {
        public IActionResult Index() {
            return View("Message", "This is the Index action on the Home controller");
        }

        public IActionResult Secure() {
            return View("Message", "This is the Secure action on the Home controller");
        }
    }
}
```
x??

---

#### Understanding the `AuthorizationFilterContext` Class
Background context: The `AuthorizationFilterContext` class provides additional information about the current authorization filter execution, such as whether to continue or interrupt the request processing based on the validation results.

:p What properties are included in the `AuthorizationFilterContext` object?
??x
The `AuthorizationFilterContext` object includes several properties:
- `Result`: An `IActionResult` that can be set to interrupt further filter execution and return a result directly.
- Other properties like `ActionDescriptor`, `HttpContext`, etc., provide details about the request and response.

Example property usage:
```csharp
public void OnAuthorization(AuthorizationFilterContext context) {
    if (!context.HttpContext.Request.IsHttps) {
        // Interrupt further execution by setting the Result
        context.Result = new StatusCodeResult(StatusCodes.Status403Forbidden);
    }
}
```
x??

---

#### Resource Filters in ASP.NET Core
Background context: In ASP.NET Core, resource filters are used to provide additional functionality before and after an action is executed. These filters can be asynchronous or synchronous and offer opportunities for developers to customize the request processing pipeline.

:p What are resource filters and how do they work in ASP.NET Core?
??x
Resource filters allow developers to execute custom logic at specific points during the request handling process in ASP.NET Core applications. They are particularly useful when you need to implement features like data caching, logging, or authentication checks.

For synchronous resource filters, these methods are executed twice: once before model binding and again before the action result is processed. For asynchronous filters, a delegate is provided to handle execution asynchronously.

Here’s an example of how a simple cache filter works:

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace WebApp.Filters
{
    public class SimpleCacheAttribute : Attribute, IResourceFilter
    {
        private Dictionary<PathString, IActionResult> CachedResponses = new Dictionary<PathString, IActionResult>();

        public void OnResourceExecuting(ResourceExecutingContext context)
        {
            PathString path = context.HttpContext.Request.Path;
            if (CachedResponses.ContainsKey(path))
            {
                context.Result = CachedResponses[path];
                CachedResponses.Remove(path);
            }
        }

        public void OnResourceExecuted(ResourceExecutedContext context)
        {
            if (context.Result != null)
            {
                CachedResponses.Add(context.HttpContext.Request.Path, context.Result);
            }
        }
    }
}
```

In this example, the `OnResourceExecuting` method checks for cached results and returns them to short-circuit the pipeline. The `OnResourceExecuted` method caches new action results.
x??

---
#### Asynchronous Resource Filters
Background context: Asynchronous resource filters provide a way to handle requests asynchronously before an action is executed or after it has been handled but before the action result is processed.

:p What are asynchronous resource filters, and how do they differ from synchronous ones?
??x
Asynchronous resource filters allow developers to execute custom logic in an asynchronous manner. They use the `IAsyncResourceFilter` interface instead of `IResourceFilter`. The key difference lies in their handling through a delegate that is provided for invoking further processing.

Here’s how you can implement an asynchronous cache filter:

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;

namespace WebApp.Filters
{
    public class SimpleCacheAttribute : Attribute, IAsyncResourceFilter
    {
        private Dictionary<PathString, IActionResult> CachedResponses = new Dictionary<PathString, IActionResult>();

        public async Task OnResourceExecutionAsync(ResourceExecutingContext context, ResourceExecutionDelegate next)
        {
            PathString path = context.HttpContext.Request.Path;
            if (CachedResponses.ContainsKey(path))
            {
                context.Result = CachedResponses[path];
                await next(context);
            }
            else
            {
                await next(context);
                if (context.Result != null)
                {
                    CachedResponses.Add(context.HttpContext.Request.Path, context.Result);
                }
            }
        }
    }
}
```

In this example, the `OnResourceExecutionAsync` method checks for cached results and returns them to short-circuit the pipeline or proceeds with further processing.
x??

---
#### Applying Custom Filters
Background context: Custom filters can be applied directly to controllers, actions, or Razor Pages using attributes. These filters provide a way to extend functionality without altering the core logic of the application.

:p How can custom resource filters be applied to Razor Pages?
??x
Custom resource filters can be applied to Razor Pages by decorating the page model with the filter attribute. Here’s how you can apply a caching filter to a Razor Page:

```csharp
@page "/pages/message"
@model MessageModel
@using Microsoft.AspNetCore.Mvc.RazorPages
@using WebApp.Filters

@if (Model.Message is string)
{
    @Model.Message
}
else if (Model.Message is IDictionary<string, string>)
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

@functions
{
    [RequireHttps]
    [SimpleCache]
    public class MessageModel : PageModel
    {
        public object Message { get; set; } = 
            DateTime.Now.ToLongTimeString() + " This is the Message Razor Page";
    }
}
```

In this example, the `[SimpleCache]` attribute is used to apply a caching filter to the `MessageModel`. The `RequireHttps` attribute ensures that HTTPS is required for accessing the page.
x??

---

---
#### Resource Execution Context and Caching
Background context explaining how resource filters work, including short-circuiting the pipeline. Describe the purpose of caching responses and how it impacts performance.

:p How does the `OnResourceExecutionAsync` method handle cached responses?
??x
The `OnResourceExecutionAsync` method checks if a cached response exists for the given path. If found, it directly sets the result to avoid further processing. Otherwise, it proceeds to execute the resource and caches the result after handling the request.

```csharp
if (CachedResponses.ContainsKey(path)) {
    context.Result = CachedResponses[path];
    CachedResponses.Remove(path);
} else {
    ResourceExecutedContext execContext = await next();
    if (execContext.Result != null) { // Ensure not null before caching
        CachedResponses.Add(context.HttpContext.Request.Path, execContext.Result);
    }
}
```
x??

---
#### Action Filters vs. Resource Filters
Explain the differences between action filters and resource filters in terms of execution timing and their respective scopes.

:p What is the primary difference between action filters and resource filters?
??x
The primary difference lies in when they are executed and their scope:
- **Resource Filters**: Executed before model binding (short-circuiting possible). Can be applied to Razor Pages.
- **Action Filters**: Executed after model binding, specifically for controllers and actions. Not applicable to Razor Pages.

Example of resource filter short-circuiting:
```csharp
public class MyResourceFilter : IResourceFilter {
    public void OnResourceExecuting(ResourceExecutingContext context) {
        // Short-circuit logic here
    }

    public void OnResourceExecuted(ResourceExecutedContext context) {
        // Post-execution logic here
    }
}
```

Example of action filter:
```csharp
public class MyActionFilter : IActionFilter {
    public void OnActionExecuting(ActionExecutingContext context) {
        // Logic before action execution
    }

    public void OnActionExecuted(ActionExecutedContext context) {
        // Logic after action execution
    }
}
```
x??

---
#### ActionExecutingContext Properties
Detail the properties available in `ActionExecutingContext` that provide information about the upcoming method call.

:p What are some key properties of the `ActionExecutingContext`?
??x
Key properties include:
- **Controller**: Returns the controller whose action is about to be invoked.
- **ActionArguments**: A dictionary of arguments passed to the action, indexed by name. Other properties like FilterContext for additional context.

Example usage within an action filter:
```csharp
public class MyActionFilter : IActionFilter {
    public void OnActionExecuting(ActionExecutingContext context) {
        var controller = context.Controller;
        var args = context.ActionArguments;
    }
}
```
x??

---
#### IActionFilter Interface
Explain the purpose and methods of `IActionFilter` and how they are used in the pipeline.

:p What does the `IActionFilter` interface define?
??x
The `IActionFilter` interface defines two main methods for managing actions:
- **OnActionExecuting(ActionExecutingContext context)**: Called just before an action method is executed. Useful for pre-processing.
- **OnActionExecuted(ActionExecutedContext context)**: Called after the action method has been executed. Use this to clean up or post-process.

Example of implementing `IActionFilter`:
```csharp
public class MyActionFilter : IActionFilter {
    public void OnActionExecuting(ActionExecutingContext context) {
        // Pre-action logic here
    }

    public void OnActionExecuted(ActionExecutedContext context) {
        // Post-action logic here
    }
}
```
x??

---

---
#### ActionExecutedContext Properties
Background context explaining that `ActionExecutedContext` is used to represent an action after it has been executed. It includes properties such as Controller, Canceled, Exception, ExceptionDispatchInfo, ExceptionHandled, and Result.

:p What property of `ActionExecutedContext` returns the controller object whose action method will be invoked?
??x
The `Controller` property returns the controller object.
x??

---
#### Canceled Property in ActionExecutedContext
Background context explaining that the `Canceled` property is set to true if another filter has short-circuited the pipeline by assigning an action result.

:p What does the `Canceled` property indicate?
??x
The `Canceled` property indicates whether another filter has short-circuited the pipeline by setting this property to true.
x??

---
#### Exception Property in ActionExecutedContext
Background context explaining that the `Exception` property contains any exception thrown by the action method.

:p What does the `Exception` property contain?
??x
The `Exception` property contains any exception that was thrown by the action method.
x??

---
#### ExceptionDispatchInfo Property in ActionExecutedContext
Background context explaining that the `ExceptionDispatchInfo` property provides details of any exception thrown by the action method.

:p What does the `ExceptionDispatchInfo` property return?
??x
The `ExceptionDispatchInfo` property returns an object containing the stack trace details of any exception thrown by the action method.
x??

---
#### ExceptionHandled Property in ActionExecutedContext
Background context explaining that the `ExceptionHandled` property indicates if the filter has handled the exception.

:p What does setting the `ExceptionHandled` property to true indicate?
??x
Setting the `ExceptionHandled` property to true indicates that the filter has handled the exception, preventing it from being propagated further.
x??

---
#### Result Property in ActionExecutedContext
Background context explaining that the `Result` property returns the result of the action method.

:p What does the `Result` property return?
??x
The `Result` property returns the `IActionResult` produced by the action method.
x??

---
#### Asynchronous Action Filters and IAsyncActionFilter Interface
Background context explaining that asynchronous action filters implement the `IAsyncActionFilter` interface, which includes an `OnActionExecutionAsync` method.

:p What is the purpose of implementing the `IAsyncActionFilter` interface?
??x
Implementing the `IAsyncActionFilter` interface allows for creating asynchronous action filters by defining the `OnActionExecutionAsync` method.
x??

---
#### IAsyncActionFilter Interface Implementation
Background context explaining that the `ChangeArgAttribute` class implements the `IAsyncActionFilter` interface and changes an argument value.

:p How does the `ChangeArgAttribute` implement the `IAsyncActionFilter` interface?
??x
The `ChangeArgAttribute` class implements the `IAsyncActionFilter` interface by defining the `OnActionExecutionAsync` method. This method checks for the presence of "message1" in the action arguments and changes its value to "New message".

```csharp
public class ChangeArgAttribute : Attribute, IAsyncActionFilter {
    public async Task OnActionExecutionAsync(
        ActionExecutingContext context,
        ActionExecutionDelegate next) {
        if (context.ActionArguments.ContainsKey("message1")) {
            context.ActionArguments["message1"] = "New message";
        }
        await next();
    }
}
```
x??

---

#### Applying Filters to Action Methods
Background context: The provided text explains how to apply filters to action methods in an ASP.NET Core application. This involves using custom attributes and understanding how these can modify the behavior of actions, such as changing argument values or enforcing security policies.

:p What is a filter in the context of ASP.NET Core?
??x
A filter in ASP.NET Core modifies the behavior of action methods before or after they execute. Filters are applied to controllers or specific actions using attributes.
x??

---
#### Custom Filter Example: HttpsOnly Attribute
Background context: The text provides an example of applying a custom `HttpsOnly` attribute to a controller to enforce HTTPS for certain actions.

:p How does the `HttpsOnly` filter work in Listing 30.22?
??x
The `HttpsOnly` filter is applied to the `HomeController`, ensuring that any action within this controller runs only over HTTPS. If the request is not secure, it might redirect or respond with an error.
x??

---
#### Using Action Filters with Attributes
Background context: The text demonstrates how to create and apply custom filters using attributes in ASP.NET Core.

:p How can you implement a custom filter by deriving from `ActionFilterAttribute`?
??x
You can derive your custom filter class, like `ChangeArgAttribute`, from `ActionFilterAttribute`. This allows you to override the necessary methods such as `OnActionExecutionAsync` to modify action behavior.
```csharp
using Microsoft.AspNetCore.Mvc.Filters;

namespace WebApp.Filters
{
    public class ChangeArgAttribute : ActionFilterAttribute
    {
        public override async Task OnActionExecutionAsync(
            ActionExecutingContext context, 
            ActionExecutionDelegate next)
        {
            if (context.ActionArguments.ContainsKey("message1"))
            {
                context.ActionArguments["message1"] = "New message";
            }
            await next();
        }
    }
}
```
x??

---
#### Implementing Filters Directly in Controllers
Background context: The text shows how to directly implement filter functionality within a controller by overriding the `OnActionExecuting` method.

:p How can you apply filter logic directly within a controller?
??x
You can override methods like `OnActionExecuting` in your controller class to add custom filter behavior. This allows modifying action arguments or executing pre-action logic.
```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Filters;
using Microsoft.AspNetCore.Mvc.Filters;

namespace WebApp.Controllers
{
    [HttpsOnly]
    public class HomeController : Controller
    {
        // ...

        public override void OnActionExecuting(ActionExecutingContext context)
        {
            if (context.ActionArguments.ContainsKey("message1"))
            {
                context.ActionArguments["message1"] = "New message";
            }
        }
    }
}
```
x??

---
#### Difference Between Custom and Base Class Filters
Background context: The text explains the difference between creating custom filters from scratch and using base class attributes like `ActionFilterAttribute`.

:p What is the difference between a custom filter implementation and one that uses a base class?
??x
A custom filter can be created directly by implementing interfaces (`IActionFilter` or `IAsyncActionFilter`) and overriding methods. Alternatively, you can use the `ActionFilterAttribute` base class to inherit commonly used logic and then override only what is necessary.
```
// Custom implementation
public class CustomFilter : IActionFilter
{
    // Implement required methods
}

// Base class implementation
public class BaseClassFilter : ActionFilterAttribute
{
    public override void OnActionExecuting(ActionExecutingContext context)
    {
        // Custom logic
    }
}
```
x??

---

