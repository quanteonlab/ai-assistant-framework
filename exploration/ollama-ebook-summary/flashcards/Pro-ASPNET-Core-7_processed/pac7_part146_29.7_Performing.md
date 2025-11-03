# Flashcards: Pro-ASPNET-Core-7_processed (Part 146)

**Starting Chapter:** 29.7 Performing remote validation

---

#### Performing Remote Validation
Background context explaining that remote validation is a technique where client-side JavaScript performs validation checks, but these checks involve an asynchronous HTTP request to the server. This approach leverages both client-side and server-side validations to ensure robustness while maintaining a responsive user experience.

:p What is the purpose of using remote validation in web applications?
??x
Remote validation allows for client-side validation with server-side enforcement by performing asynchronous HTTP requests, ensuring that certain validations are not solely dependent on the client's environment. This approach provides both immediate feedback and accurate data integrity checks.
x??

---
#### Remote Validation vs Client-Side and Server-Side Validation
Background context explaining how remote validation strikes a balance between client-side and server-side validations. It uses client-side JavaScript for immediate feedback but leverages the server to check if certain values are valid, reducing the load on the server for other tasks.

:p How does remote validation improve user experience compared to traditional server-side validation?
??x
Remote validation improves user experience by providing instantaneous feedback without requiring a full form submission. This makes the application more responsive and reduces waiting time, especially over slower network connections.
x??

---
#### Creating Validation Controller in ASP.NET Core
Background context detailing how to create a controller for performing remote validation checks. The controller defines action methods that handle HTTP GET requests, perform data validations, and return JSON responses indicating whether the value is valid.

:p How do you set up an ASP.NET Core controller to support remote validation?
??x
You need to create a controller decorated with `[ApiController]` and `[Route("api/[controller]")].` Inside this controller, define action methods that accept parameters matching the field names for which validations are required. These methods should return `bool` values as JSON responses indicating if the input is valid.

```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    [ApiController]
    [Route("api/[controller]")]
    public class ValidationController : ControllerBase {
        private DataContext dataContext;

        public ValidationController(DataContext context) {
            dataContext = context;
        }

        [HttpGet("categorykey")]
        public bool CategoryKey(string categoryId) {
            long keyVal;
            return long.TryParse(categoryId, out keyVal)
                && dataContext.Categories.Find(keyVal) != null;
        }

        [HttpGet("supplierkey")]
        public bool SupplierKey(string supplierId) {
            long keyVal;
            return long.TryParse(supplierId, out keyVal)
                && dataContext.Suppliers.Find(keyVal) != null;
        }
    }
}
```
x??

---
#### Applying the Remote Validation Attribute
Background context explaining how to use the `Remote` attribute in ASP.NET Core models. This attribute helps integrate server-side validation checks into the client-side validation process, ensuring that both front-end and back-end validations are consistent.

:p How do you apply remote validation attributes to model properties?
??x
To apply remote validation, use the `Remote` attribute on your model property. The attribute requires specifying the action method name in the controller that will perform the validation check and the controller name itself.

```csharp
using System.ComponentModel.DataAnnotations;
namespace WebApp.Models {
    public class Product {
        [Required(ErrorMessage = "Please enter a name")]
        public required string Name { get; set; }
        
        [Remote("CategoryKey", "Validation", 
            ErrorMessage = "Enter an existing key")]
        public long CategoryId { get; set; }

        [Remote("SupplierKey", "Validation",
            ErrorMessage = "Enter an existing key")]
        public long SupplierId { get; set; }
    }
}
```
x??

---
#### Testing Remote Validation
Background context explaining how to test the remote validation setup by entering invalid data and submitting a form. This involves seeing error messages appear as input is entered, providing immediate feedback.

:p How do you test if remote validation works correctly?
??x
To test remote validation, enter an invalid key value in the relevant fields (e.g., CategoryId or SupplierId) and submit the form. You should see error messages displayed immediately upon entering incorrect values, demonstrating that client-side JavaScript is correctly triggering server-side validations.

Given the example with valid keys 1, 2, and 3:
- Enter any other number and observe the error message.
x??

---

---
#### Remote Validation Mechanism
Background context explaining how remote validation works in Razor Pages and ASP.NET Core. It involves sending HTTP requests to a server endpoint for validation, which can be triggered by user input or form submissions.

:p How does remote validation work in Razor Pages?
??x
Remote validation in Razor Pages sends asynchronous HTTP requests to a specified URL for data validation. The controller action receives the input values and performs necessary checks (e.g., database queries). This is useful for ensuring that entered data meets certain criteria before being processed by the server.

For example, consider a scenario where you have a product form with a category selection field. When a user types or changes the category ID, an HTTP request is sent to validate if this key exists in the database.
```csharp
[HttpGet("categorykey")]
public bool CategoryKey(string? categoryId, [FromQuery] KeyTarget target) {
    long keyVal;
    return long.TryParse(categoryId ?? target.CategoryId, out keyVal)
        && dataContext.Categories.Find(keyVal) != null;
}
```
x??

---
#### Customizing Validation Action Methods
Background context on how to customize validation action methods in the `ValidationController` for both Razor Pages and regular controllers. It highlights the use of model binding features to handle different types of requests.

:p How can you make a single validation action method compatible with both Razor Page and controller validation?
??x
To achieve this, you add parameters that accept both types of request in the validation action methods. This is done by using the `FromQuery` parameter attribute from ASP.NET Core's model binding features. Here’s an example:

```csharp
[HttpGet("categorykey")]
public bool CategoryKey(string? categoryId, [FromQuery] KeyTarget target) {
    long keyVal;
    return long.TryParse(categoryId ?? target.CategoryId, out keyVal)
        && dataContext.Categories.Find(keyVal) != null;
}
```
This method can handle validation requests from both the controller and Razor Pages by accepting `categoryId` or `target.CategoryId`.

For a Razor Page request:
- URL: `http://localhost:5000/api/Validation/categorykey? Product.CategoryId=1`

And for a regular controller request:
- URL: `http://localhost:5000/api/Validation/categorykey? CategoryId=1`
x??

---
#### KeyTarget Class Configuration
Background context on configuring the `KeyTarget` class to handle different types of remote validation requests. This ensures that both controller and Razor Page validations can be managed through a single method.

:p How does the `KeyTarget` class facilitate handling both types of validation requests?
??x
The `KeyTarget` class is configured using the `[Bind(Prefix = "Product")]` attribute to bind its properties to the `Product` part of the request. This allows it to match with both controller and Razor Page validation scenarios.

For example, when a validation action method receives a request, it checks for values in the `categoryId` parameter or falls back to `target.CategoryId`, depending on the type of request:
```csharp
public class KeyTarget {
    public string? CategoryId { get; set; }
    public string? SupplierId { get; set; }
}
```
This setup ensures that a single action method can handle validation requests from both sources by checking for values in either `categoryId` or `target.CategoryId`.
x??

---
#### Validation Controller Example
Background context on creating and configuring the `ValidationController` to perform remote validation. This includes setting up HTTP GET endpoints with parameters to validate data.

:p What is an example of a `ValidationController` that performs remote validation?
??x
An example of a `ValidationController` can be seen in Listing 29.27, where it includes methods for validating category and supplier keys. The controller uses model binding features to handle both types of requests:

```csharp
[ApiController]
[Route("api/[controller]")]
public class ValidationController : ControllerBase {
    private DataContext dataContext;

    public ValidationController(DataContext context) {
        dataContext = context;
    }

    [HttpGet("categorykey")]
    public bool CategoryKey(string? categoryId, [FromQuery] KeyTarget target) {
        long keyVal;
        return long.TryParse(categoryId ?? target.CategoryId, out keyVal)
            && dataContext.Categories.Find(keyVal) != null;
    }

    [HttpGet("supplierkey")]
    public bool SupplierKey(string? supplierId, [FromQuery] KeyTarget target) {
        long keyVal;
        return long.TryParse(supplierId ?? target.SupplierId, out keyVal)
            && dataContext.Suppliers.Find(keyVal) != null;
    }
}

[Bind(Prefix = "Product")]
public class KeyTarget {
    public string? CategoryId { get; set; }
    public string? SupplierId { get; set; }
}
```
This example shows how to configure the controller and `KeyTarget` class to manage both types of validation requests effectively.
x??

---

#### Filters in ASP.NET Core
Background context: In ASP.NET Core, filters are middleware components that provide a mechanism for executing custom logic at specific stages of request processing. They can be used to manage security policies, resource policies (like caching), altering requests or responses, inspecting results and exceptions, and more.

Filters are like layers in a sandwich where each layer performs a specific task before passing the request to the next layer or directly to the action handler. Filters can operate on different stages of request processing such as before or after actions, pages, result execution, and even exception handling.
:p What is the purpose of filters in ASP.NET Core?
??x
Filters are used to add custom logic at specific points during the request processing lifecycle in ASP.NET Core applications. They provide a structured way to implement features like security policies, caching, logging, and more without modifying action methods or handlers directly.

For example, you can use an authorization filter to check if a user is authorized before executing an action method.
??x
---

#### Implementing Security Policies with Filters
Background context: Authorization filters are used to enforce access control rules based on the authenticated user's identity. They typically run at the beginning of the request pipeline and decide whether to allow or deny execution of the associated action.

The `IAuthorizationFilter` interface is implemented by classes that want to perform authorization tasks.
:p How do you implement a security policy using filters in ASP.NET Core?
??x
To implement a security policy, create a class that implements the `IAuthorizationFilter` interface and override its methods. For example:

```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class CustomAuthorizationFilter : IAuthorizationFilter
{
    public void OnAuthorization(AuthorizationFilterContext context)
    {
        // Check if the user is authenticated or has specific roles
        var user = context.HttpContext.User;
        if (!user.Identity.IsAuthenticated || !user.IsInRole("Admin"))
        {
            context.Result = new ChallengeResult(); // Redirect to login page
        }
    }
}
```
The `OnAuthorization` method checks whether the current user is authenticated and has the required role. If not, it redirects the user to a login page.
??x
---

#### Implementing Resource Policies with Filters
Background context: Resource filters can be used to manage non-security related policies such as caching or logging. They execute at specific points in the request pipeline that are outside of action methods but still before or after the execution.

Resource filters have access to `IActionResult` and can return a new result, modify the current one, or let it pass through.
:p How do you implement a resource policy with filters in ASP.NET Core?
??x
To implement a resource policy, create a class that implements the `IResourceFilter` interface. For example:

```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class CachingFilter : IResourceFilter
{
    public void OnResourceExecuting(ResourceFilterContext context)
    {
        // Cache the result if certain conditions are met
        var request = context.HttpContext.Request;
        if (request.Headers["Cache-Control"] == "cache")
        {
            context.Result = new ContentResult { ContentType = "text/plain", Content = "Cached content" };
        }
    }

    public void OnResourceExecuted(ResourceFilterContext context)
    {
        // Log the result after execution
        var logMessage = $"Result: {context.Result}";
        // Log to a file or database
    }
}
```
The `OnResourceExecuting` method checks if the request has a specific header indicating caching is needed. If so, it sets the response content and type. The `OnResourceExecuted` method logs the result after execution.
??x
---

#### Altering Requests or Responses with Action Filters
Background context: Action filters are used to modify the behavior of action methods by executing custom logic before and/or after their execution.

Action filters can be applied globally using attributes, or registered in the service collection for more granular control over when they should execute.
:p How do you implement an action filter that alters requests or responses?
??x
To create an action filter that modifies requests or responses, implement the `IActionFilter` interface. For example:

```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class LoggingActionFilter : IActionFilter
{
    public void OnActionExecuting(ActionExecutingContext context)
    {
        // Log before executing the action method
        var logMessage = $"Action {context.ActionDescriptor.DisplayName} is about to execute.";
        // Log to a file or database
    }

    public void OnActionExecuted(ActionExecutedContext context)
    {
        // Log after executing the action method
        var logMessage = $"Action {context.ActionDescriptor.DisplayName} executed with result: {context.Result}";
        // Log to a file or database
    }
}
```
The `OnActionExecuting` method logs before the action method is about to execute, and `OnActionExecuted` logs after the action method has completed. This helps in monitoring and debugging.
??x
---

#### Inspecting or Altering Uncaught Exceptions with Exception Filters
Background context: Exception filters provide a mechanism to handle uncaught exceptions that may occur during request processing. They are useful for logging, custom error pages, or redirecting users based on the type of exception.

Exception filters can be used by implementing `IExceptionFilter` and handling specific types of exceptions.
:p How do you implement an exception filter in ASP.NET Core?
??x
To implement an exception filter, create a class that implements the `IExceptionFilter` interface. For example:

```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class CustomExceptionFilter : IExceptionFilter
{
    public void OnException(ExceptionContext context)
    {
        // Log the exception
        var logMessage = $"Exception: {context.Exception.Message}";
        // Log to a file or database

        // Optionally, change the result of the response
        if (context.Exception is UnauthorizedAccessException)
        {
            context.Result = new ContentResult { ContentType = "text/plain", Content = "You are not authorized" };
            context.HttpContext.Response.StatusCode = 401;
        }
    }
}
```
The `OnException` method handles any uncaught exception that occurs during request processing. It logs the exception and can modify the response to return a custom message or status code.
??x
---

#### Managing Filter Lifecycle with IOrderedFilter
Background context: The order in which filters are executed is crucial for ensuring that dependencies between them are met. `IOrderedFilter` allows you to specify the execution order of filters by implementing the `Order` property.

Setting the `Order` property ensures that filters are applied in a specific sequence, preventing issues where one filter might overwrite or interfere with another.
:p How do you manage the lifecycle and execution order of filters using IOrderedFilter?
??x
To manage the lifecycle and execution order of filters, implement the `IOrderedFilter` interface and set the `Order` property. For example:

```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class CustomActionFilter : IOrderedFilter
{
    public int Order { get; set; } = -10;

    public void OnActionExecuting(ActionExecutingContext context)
    {
        // Logic for before action execution
    }

    public void OnActionExecuted(ActionExecutedContext context)
    {
        // Logic for after action execution
    }
}
```
The `Order` property is used to determine the sequence of filter execution. A lower number means earlier execution. This ensures that filters are applied in a specific order, managing dependencies and preventing conflicts.
??x
---

