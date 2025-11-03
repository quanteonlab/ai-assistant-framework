# Flashcards: Pro-ASPNET-Core-7_processed (Part 147)

**Starting Chapter:** 30.1.3 Running the example application

---

---
#### Enabling HTTPS Connections
Background context: This section explains how to enable secure connections (HTTPS) for your .NET Core application. It is crucial for using some examples that require SSL encryption, such as sending sensitive data over the internet.

:p How do you configure HTTPS in `launchSettings.json`?
??x
To enable HTTPS and set a specific port, you need to modify the `launchSettings.json` file located in the `Properties` folder. Here is an example of how it should be configured:

```json
{
  "iisSettings": {
    "windowsAuthentication": false,
    "anonymousAuthentication": true,
    "iisExpress": {
      "applicationUrl": "http://localhost:5000",
      "sslPort": 0
    }
  },
  "profiles": {
    "WebApp": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": false,
      "applicationUrl": "http://localhost:5000;https://localhost:44350",
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    },
    "IIS Express": {
      "commandName": "IISExpress",
      "launchBrowser": true,
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    }
  }
}
```

This configuration sets up both HTTP and HTTPS endpoints, with the SSL port set to `44350`.

x??
---

#### Generating Test Certificates
Background context: .NET Core comes with a test certificate that can be used for development purposes. However, you need to regenerate it and trust it before using HTTPS in your application.

:p How do you regenerate and trust the test certificate?
??x
You can regenerate and trust the test certificate by running the following commands in the WebApp folder:

```bash
dotnet dev-certs https --clean
dotnet dev-certs https --trust
```

The `--clean` command deletes any existing trusted development certificates, while the `--trust` command regenerates a new one and adds it to your local trusted certificate store. You will be prompted with a message asking if you want to delete the existing certificate; respond "Yes" to continue.

x??
---

#### Configuring Program.cs
Background context: This section details how to configure the `Program.cs` file for your .NET Core application, specifically enabling default controller routes and removing unnecessary services and components from earlier chapters.

:p How does the configuration in `Program.cs` differ from previous versions?
??x
The configuration in `Program.cs` now uses the default controller routes and removes some of the previously used services/components. Here is how it looks:

```csharp
using Microsoft.EntityFrameworkCore;
using WebApp.Models;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddDbContext<DataContext>(opts => {
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:ProductConnection"]);
    opts.EnableSensitiveDataLogging(true);
});

builder.Services.AddControllersWithViews();
builder.Services.AddRazorPages();

var app = builder.Build();

app.UseStaticFiles();
app.MapDefaultControllerRoute();
app.MapRazorPages();

var context = app.Services.CreateScope().ServiceProvider
.GetRequiredService<DataContext>();

SeedData.SeedDatabase(context);

app.Run();
```

This configuration sets up the database connection, enables sensitive data logging for development purposes, and maps the default routes for controllers and Razor pages.

x??
---

#### Dropping the Database
Background context: This section explains how to drop the database before running the application. Dropping the database ensures that any existing data is removed so that a fresh state can be initialized upon running the application.

:p How do you drop the database using PowerShell?
??x
To drop the database, open a new PowerShell command prompt and navigate to the folder containing the `WebApp.csproj` file. Then, run the following command:

```powershell
dotnet ef dbcontext drop --force
```

This command will delete the existing database schema. The `--force` flag ensures that any data in the database is also dropped.

x??
---

#### Using Filters in ASP.NET Core
Background context: In ASP.NET Core, filters allow you to apply cross-cutting concerns (like logging or authorization) at a higher level than individual action methods. This makes your controllers cleaner and more maintainable.

:p What are filters in the context of ASP.NET Core?
??x
Filters in ASP.NET Core enable you to define logic that can be applied across multiple actions, reducing code duplication and improving maintainability. For example, you can enforce HTTPS for specific actions or controllers without cluttering each action method.
x??

---
#### Enforcing HTTPS Using Action Methods
Background context: You can directly enforce HTTPS checks within action methods, but this approach has limitations in terms of scalability and maintainability.

:p How does enforcing HTTPS directly within an action method work?
??x
Enforcing HTTPS directly within an action method involves checking the `Request.IsHttps` property to determine if the request is secure. If not, you can return a forbidden status code or redirect the user to a secure URL.
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    public class HomeController : Controller {
        public IActionResult Index() {
            if (Request.IsHttps) {
                return View("Message", "This is the Index action on the Home controller");
            } else {
                return new StatusCodeResult(StatusCodes.Status403Forbidden);
            }
        }
    }
}
```
x??

---
#### Limitations of Direct HTTPS Enforcement
Background context: While directly enforcing HTTPS within actions can work, it has several limitations that make it less desirable for large applications.

:p What are the main drawbacks of implementing HTTPS enforcement directly in action methods?
??x
The main drawbacks include:
1. **Code Duplication**: You need to repeat the same check across multiple action methods and controllers.
2. **Maintenance Issues**: As new actions or controllers are added, you must remember to add the HTTPS check, which can lead to security vulnerabilities if forgotten.

Example of code duplication:
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    public class HomeController : Controller {
        public IActionResult Index() { ... }
        public IActionResult Secure() { ... } // Repeats the same logic
    }
}
```
x??

---
#### Using Filters to Enforce HTTPS
Background context: Filters provide a more maintainable way to enforce rules like requiring HTTPS, by centralizing this logic.

:p How can filters be used to enforce HTTPS in ASP.NET Core?
??x
Filters can be created to handle cross-cutting concerns such as enforcing HTTPS. By defining a custom filter, you can apply the HTTPS check once and have it automatically applied to all actions that need it.
```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class SecureActionFilter : Attribute, IActionFilter {
    public void OnActionExecuting(ActionExecutingContext context) {
        if (!context.HttpContext.Request.IsHttps) {
            context.Result = new StatusCodeResult(StatusCodes.Status403Forbidden);
        }
    }

    public void OnActionExecuted(ActionExecutedContext context) { }
}
```
x??

---
#### Applying Filters to Controllers or Actions
Background context: You can apply filters globally, per controller, or even per action method.

:p How do you apply a filter to a specific action method?
??x
You can apply a filter to a specific action method by decorating the action method with the `[ServiceFilter]` attribute.
```csharp
using Microsoft.AspNetCore.Mvc;

[SecureActionFilter]
public class HomeController : Controller {
    public IActionResult Index() { ... }
}
```
Alternatively, you can also apply filters globally or per controller:
```csharp
services.AddControllers(options => options.Filters.Add(typeof(SecureActionFilter)));
```
x??

---

#### Applying Filters to Controllers

Background context: The provided text discusses how to apply filters, specifically `RequireHttps`, to restrict access to action methods so that only HTTPS requests are supported. This is useful for enhancing security but requires careful consideration during development.

:p How does the `RequireHttps` attribute work in ASP.NET Core controllers?
??x
The `RequireHttps` attribute ensures that all actions within a controller or specified action method can only be accessed via HTTPS. If an HTTP request is made, it is redirected to its corresponding HTTPS URL. This helps maintain security but needs adjustment during development where HTTP and HTTPS might use different ports.

Code example:
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    public class HomeController : Controller {
        [RequireHttps]
        public IActionResult Index() {
            return View("Message", "This is the Index action on the Home controller");
        }

        [RequireHttps]
        public IActionResult Secure() {
            return View("Message", "This is the Secure action on the Home controller");
        }
    }
}
```
x??

---
#### Applying Filters to Entire Controller Classes

Background context: The text explains that applying `RequireHttps` at the class level affects all actions within that controller. This simplifies security implementation but still requires awareness of potential redirection issues during development.

:p How can the `RequireHttps` attribute be applied to an entire controller class?
??x
By applying the `RequireHttps` attribute to a controller class, it ensures that every action method in that class will require HTTPS requests. However, this approach means remembering to add the attribute, which could lead to oversights.

Code example:
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers {
    [RequireHttps]
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
#### Customizing Filter Behavior

Background context: The `RequireHttps` attribute includes a protected method called `HandleNonHttpsRequest`, which can be overridden to customize behavior. This provides flexibility in handling non-HTTPS requests, such as during development.

:p What is the purpose of overriding `HandleNonHttpsRequest` in the `RequireHttpsAttribute`?
??x
Overriding `HandleNonHttpsRequest` allows customization of how non-HTTPS requests are handled when using the `RequireHttps` attribute. This can be particularly useful for development environments where HTTP and HTTPS might use different local ports.

Code example:
```csharp
public class CustomRequireHttpsAttribute : RequireHttpsAttribute {
    protected override void HandleNonHttpsRequest(AuthorizationFilterContext context) {
        // Custom logic here, e.g., logging or custom redirect behavior
        base.HandleNonHttpsRequest(context);
    }
}
```
x??

---
#### Using Filters in Razor Pages

Background context: The text also mentions how filters can be applied to Razor Pages. Similar to controllers, the `RequireHttps` attribute can be used on individual handler methods or at the class level.

:p How can you implement HTTPS-only policy in a Razor Page using `RequireHttps`?
??x
To implement an HTTPS-only policy in a Razor Page, you can use the `RequireHttps` attribute either on the handler method or on the entire page model. This ensures that only HTTPS requests are processed by the page.

Code example:
```csharp
@page "/pages/message"
@model MessageModel
@using Microsoft.AspNetCore.Mvc.RazorPages

[RequireHttps]
public class MessageModel : PageModel {
    public object Message { get; set; } = "This is the Message Razor Page";

    public IActionResult OnGet() {
        if (Request.IsHttps) {
            return new StatusCodeResult(StatusCodes.Status403Forbidden);
        } else {
            return Page();
        }
    }
}
```
x??

---
#### Global Filters

Background context: The text briefly mentions global filters, which can apply a filter to every action in an application. However, the specific details of implementing global filters are not provided in this excerpt.

:p What is meant by "global filters" in ASP.NET Core?
??x
Global filters are filters that apply to all actions across an entire application rather than being limited to individual controllers or pages. They can be defined in the `Startup.cs` file or through the `ConfigureServices` method.

Code example:
```csharp
public void ConfigureServices(IServiceCollection services) {
    services.AddControllers(options => options.Filters.Add(new RequireHttpsAttribute()));
}
```
x??

---

---
#### MessageModel Class Description
Background context: The provided code snippet shows a simplified version of an ASP.NET Core Razor Page Model. It demonstrates how to define and use properties, constructors, and potentially action methods within a Razor Page.

:p What is the purpose of defining `Message` as a property in the `MessageModel` class?
??x
The purpose of defining `Message` as a property is to store and provide access to a message that will be displayed on the page. Properties allow for easy data binding between the model and the Razor view, ensuring that the message can be dynamically set or accessed based on user interactions.

```csharp
public class MessageModel : PageModel {
    public object Message { get; set; } = "This is the Message Razor Page";
}
```
x??
---

---
#### Understanding Filters in ASP.NET Core
Background context: The provided text explains various types of filters supported by ASP.NET Core, their purposes, and how they interact within the application pipeline. These filters are used to enhance the functionality and security of web applications.

:p What is the main purpose of using filters in an ASP.NET Core application?
??x
The main purpose of using filters in an ASP.NET Core application is to modularize common processing tasks related to requests, responses, actions, or pages. Filters can be applied at different stages of request processing and allow developers to implement features such as authorization, caching, exception handling, and more without cluttering the controller or action methods.

x??
---

---
#### Authorization Filters
Background context: The text describes `AuthorizationFilters`, which are used to enforce security policies on a per-action or per-controller basis. These filters can short-circuit the filter pipeline if certain conditions are not met.

:p What is an authorization filter in ASP.NET Core, and how does it work?
??x
An authorization filter in ASP.NET Core is designed to apply the application’s authorization policy. If the user is unauthenticated or lacks necessary permissions, the authorization filter can prevent further processing of the request by returning an error response immediately.

```csharp
public class CustomAuthorizationFilter : IAsyncAuthorizationFilter {
    public async Task OnAuthorizationAsync(AuthorizationFilterContext context) {
        // Logic to check if user is authorized
        if (!User.IsInRole("Admin")) {
            context.Result = new ForbidResult();
            return;
        }
    }
}
```
x??
---

---
#### Resource Filters
Background context: `ResourceFilters` are used to intercept requests and can be utilized for caching, logging, or other similar purposes. They operate before the request reaches the endpoint.

:p What is a resource filter in ASP.NET Core, and when would you use it?
??x
A resource filter in ASP.NET Core is used to intercept requests before they reach the endpoint. This type of filter can be useful for implementing caching strategies where you might want to check if a cached response can be returned without hitting the database or other services.

```csharp
public class CachingFilter : IAsyncResourceFilter {
    public async Task OnResourceExecutionAsync(ResourceFilterContext context) {
        // Logic to determine if caching should occur
        if (context.HttpContext.Request.Method == "GET") {
            var cacheEntry = await GetFromCache(context.HttpContext.Request.Path);
            if (cacheEntry != null) {
                context.Result = new ContentResult { Content = cacheEntry };
            }
        }
    }
}
```
x??
---

---
#### Action Filters
Background context: `ActionFilters` are used to modify requests before they reach the action method or modify responses after an action has executed. They can be applied only to controllers and actions.

:p What is an action filter in ASP.NET Core, and what can it do?
??x
An action filter in ASP.NET Core is used to modify requests before they reach the action method or to modify the response after the action has executed. These filters are particularly useful for adding common functionality such as logging, performance monitoring, or applying cross-cutting concerns.

```csharp
public class PerformanceLoggingFilter : IActionFilter {
    public void OnActionExecuting(ActionExecutingContext context) {
        // Log before the action executes
        Console.WriteLine("Action is about to execute");
    }

    public void OnActionExecuted(ActionExecutedContext context) {
        // Log after the action executes
        Console.WriteLine("Action has executed");
    }
}
```
x??
---

---
#### Page Filters in ASP.NET Core
Background context: `PageFilters` are specifically used for Razor Pages and allow you to modify requests before they reach the handler method or alter responses after the page has been processed.

:p What is a page filter in ASP.NET Core, and what does it do?
??x
A page filter in ASP.NET Core modifies requests before they reach the handler method of a Razor Page or alters the response after the page has been processed. These filters are useful for handling common tasks like model binding or performing actions specific to a page.

```csharp
public class PageInitializationFilter : IPageFilter {
    public void OnPageInitializing(PageContext context) {
        // Initialize models, perform validations, etc.
    }

    public void OnPageInitialized(PageContext context) {
        // Additional processing after the page has initialized
    }
}
```
x??
---

---
#### Result Filters in ASP.NET Core
Background context: `ResultFilters` are used to alter the action result before it is executed or modify the result after execution. They can be applied to both controllers and Razor Pages.

:p What is a result filter in ASP.NET Core, and what tasks can it perform?
??x
A result filter in ASP.NET Core is used to alter the action result before it is executed or to modify the result after execution. These filters are useful for adding common functionality such as modifying response content, applying security headers, or performing cleanup.

```csharp
public class ResponseModificationFilter : IAsyncResultFilter {
    public void OnResultExecuting(ResultExecutingContext context) {
        // Modify the result before it is executed
    }

    public void OnResultExecuted(ResultExecutedContext context) {
        // Modify the result after it has been executed
    }
}
```
x??
---

---
#### Exception Filters in ASP.NET Core
Background context: `ExceptionFilters` are used to handle exceptions that occur during the execution of an action method or page handler. They provide a centralized way to manage errors and can return custom responses.

:p What is an exception filter in ASP.NET Core, and how does it work?
??x
An exception filter in ASP.NET Core handles exceptions that occur during the execution of an action method or page handler. These filters allow you to catch and respond to unexpected errors by returning a custom response or logging the error.

```csharp
public class CustomExceptionFilter : IExceptionFilter {
    public void OnException(ExceptionContext context) {
        // Log the exception details
        Console.WriteLine($"An error occurred: {context.Exception.Message}");

        // Optionally, handle the exception and return a custom response
        context.Result = new ContentResult { Content = "Something went wrong." };
    }
}
```
x??
---

