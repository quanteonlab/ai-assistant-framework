# Flashcards: Pro-ASPNET-Core-7_processed (Part 150)

**Starting Chapter:** 30.5 Managing the filter lifecycle

---

#### Action Method Exception Handling
Background context explaining how action methods handle exceptions and the specific behavior described. The `GenerateException` action method relies on the default routing pattern to receive a nullable int value from the request URL. It throws an `ArgumentNullException` if there is no matching URL segment and an `ArgumentOutOfRangeException` if its value exceeds 10. If the value is in range, it returns a ViewResult.

:p Describe what happens when requesting `/Home/GenerateException /100`.
??x
When you request `/Home/GenerateException /100`, the final segment (`100`) will exceed the expected range (greater than 10), causing an `ArgumentOutOfRangeException` to be thrown. Since this exception is handled by a filter, it will produce the result specified by that filter.

If you were to request only `/Home/GenerateException`, without any additional path segments, the action method would not throw an exception because there's no value for the parameter, leading to an `ArgumentNullException`.
x??

---

#### Exception Filter Lifecycle
The default behavior of ASP.NET Core is to manage filters and reuse them for subsequent requests. However, sometimes you need more control over how filters are created.

:p Explain the lifecycle of a filter by adding a custom attribute named `GuidResponseAttribute`.
??x
To create a filter that tracks its lifecycle, add a class file called `GuidResponseAttribute.cs` to the Filters folder and define the following:

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Filters;
using Microsoft.AspNetCore.Mvc.ModelBinding;
using Microsoft.AspNetCore.Mvc.ViewFeatures;

namespace WebApp.Filters {
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = true)]
    public class GuidResponseAttribute : Attribute, IAsyncAlwaysRunResultFilter {
        private int counter = 0;
        private string guid = Guid.NewGuid().ToString();

        public async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next) {
            Dictionary<string, string> resultData;

            if (context.Result is ViewResult vr && vr.ViewData.Model is Dictionary<string, string> data) {
                resultData = data;
            } else {
                resultData = new Dictionary<string, string>();
                context.Result = new ViewResult() {
                    ViewName = "/Views/Shared/Message.cshtml",
                    ViewData = new ViewDataDictionary(
                        new EmptyModelMetadataProvider(),
                        new ModelStateDictionary()
                    ) {
                        Model = resultData
                    }
                };
            }

            while (resultData.ContainsKey($"Counter_{counter}")) {
                counter++;
            }

            resultData[$"Counter_{counter}"] = guid;
            await next();
        }
    }
}
```

This attribute creates a unique GUID and appends it to the `ViewData` dictionary as a key-value pair, where each key is of the form `Counter_x`, with `x` being an incremented counter. This ensures that each filter execution results in a new message displayed.

:p How does applying this attribute twice affect the outcome?
??x
Applying the `GuidResponseAttribute` twice to the same action method or controller will result in two unique GUIDs being added to the `ViewData` dictionary, both with their own counter values. This means that if you apply the filter twice, each application will generate a new GUID and append it to the `ViewData`, ensuring that no matter how many times the attribute is applied, a distinct message is displayed.

For example:
```csharp
[GuidResponse]
[GuidResponse]
public class HomeController : Controller {
    public IActionResult Index() {
        return View("Message", "This is the Index action on the Home controller");
    }
}
```

In this case, two GUIDs will be added to the `ViewData`, each with a unique counter value.
x??

---

#### Result Filter in Action
Background context explaining how result filters can modify the outcome of an action method. The provided example demonstrates creating and applying a custom filter that replaces the original action result with one that renders a specific view and displays a unique GUID.

:p How does the `OnResultExecutionAsync` method ensure that each counter value is unique?
??x
The `OnResultExecutionAsync` method ensures uniqueness by using a while loop to find an unused key in the `resultData` dictionary. It checks if any existing keys are of the form `Counter_x`. If so, it increments `counter` and tries again until it finds a unique key.

Here is how the logic works:

1. Initialize `resultData` based on whether the current result is a `ViewResult`.
2. Use a while loop to check if the dictionary already contains a key of the form `Counter_x`.
3. Increment `counter` each time a duplicate key is found.
4. Once a unique key is identified, store the GUID with this key in `resultData`.

```csharp
while (resultData.ContainsKey($"Counter_{counter}")) {
    counter++;
}

resultData[$"Counter_{counter}"] = guid;
```

This ensures that each call to `OnResultExecutionAsync` appends a new, unique message to the `ViewData`.

:p What is the purpose of using `Dictionary<string, string>` for storing messages in the view?
??x
Using `Dictionary<string, string>` for storing messages in the view allows you to associate different keys with corresponding GUID values. Each key-value pair can represent a distinct piece of information or message that gets displayed in the view.

For example, if you have multiple filters and each one generates a unique message, using a dictionary ensures that these messages can be easily accessed by their respective keys. This makes it easy to manage and display different types of messages without overwriting existing ones.

The purpose is to provide flexibility and clarity in managing various pieces of information that need to be displayed dynamically based on the filter lifecycle.
x??

---

#### Managing Filter Lifecycle
Background context: In ASP.NET Core, filters can be used to modify or control the execution flow of an action. Filters are reusable by default, which means that a single filter instance can handle multiple requests. This behavior is useful for performance optimization but may not always align with certain scenarios where new instances are desired.

:p How does managing the lifecycle of filters in ASP.NET Core demonstrate their reusability?
??x
The lifecycle management of filters in ASP.NET Core shows how filters are created and reused across different requests. By default, filters can be reused to handle multiple requests efficiently, but sometimes it's necessary to create new instances for each request.

For example, if a filter generates unique values (like GUIDs) per request, reusing the same instance might not yield distinct results, leading to the same GUID being generated across different requests. To demonstrate this, we can set the `IsReusable` property to `false`, ensuring that a new filter instance is created for each request.

```csharp
public class GuidResponseAttribute : Attribute, IAsyncAlwaysRunResultFilter, IFilterFactory
{
    private int counter = 0;
    private string guid = Guid.NewGuid().ToString();
    
    public bool IsReusable => false; // Ensures the filter is not reusable
    
    public IFilterMetadata CreateInstance(IServiceProvider serviceProvider)
    {
        return ActivatorUtilities.GetServiceOrCreateInstance<GuidResponseAttribute>(serviceProvider);
    }
    
    public async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data)
        {
            resultData = data;
        }
        else
        {
            resultData = new Dictionary<string, string>();
            context.Result = new ViewResult()
            {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary())
                {
                    Model = resultData
                }
            };
        }
        
        while (resultData.ContainsKey($"Counter_{counter}"))
        {
            counter++;
        }
        
        resultData[$"Counter_{counter}"] = guid;
        await next();
    }
}
```
x??

---

#### Creating Filter Factories
Background context: To control how filters are created and reused, you can implement the `IFilterFactory` interface. This allows developers to specify whether a filter instance should be reusable or if new instances need to be created for each request.

:p How does implementing the `IFilterFactory` interface in ASP.NET Core affect the lifecycle of filters?
??x
Implementing the `IFilterFactory` interface in ASP.NET Core provides fine-grained control over how and when filter instances are created. By default, filters can be reused to handle multiple requests efficiently. However, sometimes it is necessary to create a new instance for each request, especially if certain properties or states need to be unique per request.

To demonstrate this, you would implement the `IFilterFactory` interface and set the `IsReusable` property to `false`, indicating that a new filter should be created for every request. This can be achieved by using the `GetServiceOrCreateInstance` method from the `ActivatorUtilities` class.

```csharp
public class GuidResponseAttribute : Attribute, IAsyncAlwaysRunResultFilter, IFilterFactory
{
    private int counter = 0;
    private string guid = Guid.NewGuid().ToString();
    
    public bool IsReusable => false; // Ensures the filter is not reusable
    
    public IFilterMetadata CreateInstance(IServiceProvider serviceProvider)
    {
        return ActivatorUtilities.GetServiceOrCreateInstance<GuidResponseAttribute>(serviceProvider);
    }
    
    public async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data)
        {
            resultData = data;
        }
        else
        {
            resultData = new Dictionary<string, string>();
            context.Result = new ViewResult()
            {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary())
                {
                    Model = resultData
                }
            };
        }
        
        while (resultData.ContainsKey($"Counter_{counter}"))
        {
            counter++;
        }
        
        resultData[$"Counter_{counter}"] = guid;
        await next();
    }
}
```
x??

---

#### Demonstrating Filter Reuse
Background context: To verify that filters are being reused, you can check the behavior of the filter instances across multiple requests. If a filter is reusable, it should produce consistent results when handling different requests. However, if a new instance needs to be created for each request, the generated values or states must change accordingly.

:p How do you confirm that filters are being reused in ASP.NET Core?
??x
To confirm that filters are being reused in ASP.NET Core, you can follow these steps:

1. Restart the ASP.NET Core application.
2. Request `https://localhost:44350/?diag`.
3. The response will contain GUID values from two `Guid-Response` filter attributes.
4. Two instances of the filter handle the request, and they produce the same GUID values.
5. Reload the browser to see if the same GUIDs are displayed again.

This indicates that the filter objects created for the first request have been reused.

To demonstrate this, you would implement a `GuidResponseAttribute` class that sets the `IsReusable` property to `true`. When the application is run, the same GUID values will be displayed in subsequent requests, indicating reuse.

```csharp
public class GuidResponseAttribute : Attribute, IAsyncAlwaysRunResultFilter, IFilterFactory
{
    private int counter = 0;
    private string guid = Guid.NewGuid().ToString();
    
    public bool IsReusable => true; // Ensures the filter is reusable
    
    public IFilterMetadata CreateInstance(IServiceProvider serviceProvider)
    {
        return ActivatorUtilities.GetServiceOrCreateInstance<GuidResponseAttribute>(serviceProvider);
    }
    
    public async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data)
        {
            resultData = data;
        }
        else
        {
            resultData = new Dictionary<string, string>();
            context.Result = new ViewResult()
            {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary())
                {
                    Model = resultData
                }
            };
        }
        
        while (resultData.ContainsKey($"Counter_{counter}"))
        {
            counter++;
        }
        
        resultData[$"Counter_{counter}"] = guid;
        await next();
    }
}
```
x??

---

#### Creating New Filter Instances Using `GetServiceOrCreateInstance`
Background context: The `IFilterFactory` interface allows you to control the creation of filter instances. By implementing this interface, you can specify that new filter instances should be created for each request using the `GetServiceOrCreateInstance` method.

:p How does the `GetServiceOrCreateInstance` method work in creating new filter instances?
??x
The `GetServiceOrCreateInstance` method is part of the `ActivatorUtilities` class and is used to create a new instance of a filter when needed. This method takes an `IServiceProvider` as input, which provides access to services registered in the application's service container.

When you implement the `IFilterFactory` interface and set the `IsReusable` property to `false`, it tells ASP.NET Core not to reuse the same instance for different requests. Instead, a new filter instance is created each time the method is called.

Here’s an example of how this can be implemented in C#:

```csharp
public class GuidResponseAttribute : Attribute, IAsyncAlwaysRunResultFilter, IFilterFactory
{
    private int counter = 0;
    private string guid = Guid.NewGuid().ToString();
    
    public bool IsReusable => false; // Ensures the filter is not reusable
    
    public IFilterMetadata CreateInstance(IServiceProvider serviceProvider)
    {
        return ActivatorUtilities.GetServiceOrCreateInstance<GuidResponseAttribute>(serviceProvider);
    }
    
    public async Task OnResultExecutionAsync(ResultExecutingContext context, ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data)
        {
            resultData = data;
        }
        else
        {
            resultData = new Dictionary<string, string>();
            context.Result = new ViewResult()
            {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary())
                {
                    Model = resultData
                }
            };
        }
        
        while (resultData.ContainsKey($"Counter_{counter}"))
        {
            counter++;
        }
        
        resultData[$"Counter_{counter}"] = guid;
        await next();
    }
}
```
x??

---
#### Registering Filters as Services
Filters can be registered as services to control their lifecycle using dependency injection. This allows for more flexible and controlled creation of filter instances per request or other scopes.
:p How are filters typically managed regarding their lifecycle in ASP.NET Core?
??x
Filters are usually created on a per-request basis by default, meaning that one instance is created per request. However, when registered as scoped services, the lifecycle can be managed more precisely using dependency injection (DI). For example, a service can be configured to create a new instance for each request or keep an existing instance alive for multiple requests.
The registration of a filter like `GuidResponseAttribute` as a scoped service is shown in Listing 30.36:
```csharp
builder.Services.AddScoped<GuidResponseAttribute>();
```
x??
---
#### Creating a Filter Service in Program.cs
This example demonstrates how to create and configure services for an ASP.NET Core application, including setting up the database context and adding controllers and Razor pages.
:p What does Listing 30.36 illustrate in terms of configuring services in `Program.cs`?
??x
Listing 30.36 illustrates creating a scoped service for a filter named `GuidResponseAttribute`. The code snippet within `Program.cs` sets up the application's dependency injection container to manage this filter’s lifecycle:
```csharp
builder.Services.AddScoped<GuidResponseAttribute>();
```
This configuration ensures that each request will get its own instance of the `GuidResponseAttribute`, providing finer control over how and when the filter is instantiated.
x??
---
#### Using Dependency Injection to Manage Filters
By managing filters as services, their lifecycles can be controlled using dependency injection. This method allows for more complex scenarios where the same service might need to create different instances based on request context or other factors.
:p How does using dependency injection affect filter management in ASP.NET Core?
??x
Using dependency injection (DI) for managing filters allows for greater flexibility and control over their lifecycle. Instead of relying on default behavior, such as creating a new instance per request, you can configure DI to create scopes that manage the creation and reuse of filter instances more effectively.
For example, if you want to ensure that each request gets a unique filter instance:
```csharp
builder.Services.AddScoped<GuidResponseAttribute>();
```
This configuration means that a separate instance will be created for each request. If you need to share an instance across multiple requests or sessions, you might configure it as a singleton instead.
x??
---
#### Applying Filters Without IFilterFactory Interface
Filters can be applied without implementing the `IFilterFactory` interface by using the `ServiceFilter` attribute. This approach allows for more straightforward and flexible application of filters to controllers or Razor Pages.
:p How can filters be applied in ASP.NET Core without implementing the `IFilterFactory` interface?
??x
Filters can be applied without implementing the `IFilterFactory` interface by utilizing the `ServiceFilter` attribute. This method simplifies the process, as it directly instructs ASP.NET Core to create and apply the filter based on its service registration.
For example:
```csharp
[ServiceFilter(typeof(GuidResponseAttribute))]
public class HomeController : Controller { ... }
```
This code snippet indicates that the `GuidResponseAttribute` should be applied to all actions in the `HomeController`. The attribute does not need to derive from `Attribute`, making it more flexible for different scenarios.
x??
---
#### Creating Global Filters
Global filters are applied to every request and do not need to be applied to individual controllers or Razor Pages. They can be configured using the options pattern, which allows adding multiple filters globally through the configuration of `MvcOptions`.
:p How does one set up a global filter in an ASP.NET Core application?
??x
Global filters are configured using the `MvcOptions.Filters` property within the `Program.cs` file. This property returns a collection where you can add filters to apply them globally.
For example, to configure the `HttpsOnlyAttribute` as a global filter:
```csharp
builder.Services.Configure<MvcOptions>(opts => 
    opts.Filters.Add<HttpsOnlyAttribute>());
```
This code snippet registers the `HttpsOnlyAttribute` so that it will be applied to every request handled by ASP.NET Core. You can add other filters in a similar manner using either the generic or non-generic `Add` methods.
x??
---

