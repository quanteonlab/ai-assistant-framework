# Flashcards: Pro-ASPNET-Core-7_processed (Part 36)

**Starting Chapter:** 12.4.3 Short-Circuiting the request pipeline

---

---
#### Class-Based Middleware Component
Class-based middleware components allow developers to define custom logic for handling HTTP requests and responses. These components can be created as classes and registered within the application pipeline using `app.UseMiddleware<ClassName>()`.
:p What is a class-based middleware component used for in ASP.NET Core?
??x
A class-based middleware component is used to add custom functionality to an ASP.NET Core application by modifying or handling HTTP requests and responses. It can be implemented as a class that defines the middleware logic, which is then registered with the application pipeline using `app.UseMiddleware<ClassName>()`.
```csharp
public class PlatformQueryStringMiddleWare : IApplicationBuilder
{
    public void Use(IApplicationBuilder next)
    {
        // Middleware logic here
    }
}
```
x??
---

---
#### Modifying Response in Return Path
Middleware components can modify the HTTP response after it has passed through other middleware or route handlers. This is achieved by calling `await context.Response.WriteAsync()` to add content to the response body.
:p How does a middleware component modify the HTTP response after processing?
??x
A middleware component modifies the HTTP response after processing by using `await context.Response.WriteAsync()` within its logic. This allows it to write additional content to the response before or after other middleware or route handlers have had their chance to process the request.
```csharp
app.Use(async (context, next) =>
{
    await next();
    if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true")
    {
        context.Response.ContentType = "text/plain";
        await context.Response.WriteAsync("Custom Middleware");
    }
    await next();
});
```
x??
---

---
#### Short-Circuiting the Request Pipeline
Components can choose not to pass the request further along the pipeline by not calling the `next()` function. This is known as short-circuiting and can be used to handle complete responses without further processing.
:p What does it mean for a middleware component to "short-circuit" the pipeline?
??x
Short-circuiting means that a middleware component handles a request completely within itself and does not pass the request along to subsequent components in the pipeline. This is achieved by not calling `next()`, which prevents further processing of the request.
```csharp
app.Use(async (context, next) =>
{
    if (context.Request.Path == "/short")
    {
        await context.Response.WriteAsync("Request Short Circuited");
    }
    else
    {
        await next();
    }
});
```
x??
---

---
#### Map Method for Creating Pipeline Branches
Background context: The `Map` method is used to create a section of middleware that processes specific URLs, enabling developers to define different sequences of middleware components based on URL patterns. This can be useful for handling requests differently depending on their paths.

:p How does the `Map` method work in creating pipeline branches?
??x
The `Map` method takes two arguments: 
1. A string representing a URL pattern.
2. An action delegate that configures a branch of middleware components to process matching URLs.

This creates a separate sequence of middleware, allowing for custom handling of specific URL paths while keeping them distinct from the main pipeline flow.

Example in C#:
```csharp
app.Map("/branch", branch => {
    branch.UseMiddleware<Platform.QueryStringMiddleWare>();
    branch.Use(async (HttpContext context, Func<Task> next) => 
    { 
        await context.Response.WriteAsync("Branch Middleware"); 
    });
});
```
x??

---
#### Branching the Request Pipeline with `MapWhen`
Background context: The `MapWhen` method allows for more flexible matching of requests based on a predicate function. This is useful when you need to route requests to different parts of the pipeline not just by URL but by other criteria as well.

:p What does the `MapWhen` method do in ASP.NET Core?
??x
The `MapWhen` method takes two arguments:
1. A predicate function that receives an `HttpContext`.
2. A function that receives an `IApplicationBuilder` object representing the pipeline branch.
This allows for routing requests based on conditions other than just URL patterns, providing a more flexible way to handle different types of requests.

Example in C#:
```csharp
app.MapWhen(ctx => ctx.Request.Path.StartsWithSegments("/special"), appBuilder =>
{
    appBuilder.UseMiddleware<Platform.QueryStringMiddleWare>();
    appBuilder.Use(async (HttpContext context, Func<Task> next) => 
    { 
        await context.Response.WriteAsync("Special Branch Middleware"); 
    });
});
```
x??

---
#### Handling Requests in Different Pipeline Paths
Background context: The `Map` method can be used to create branches within the request pipeline for specific URL patterns. These branches can have their own sequence of middleware, allowing for custom handling based on different paths.

:p How do requests flow through a pipeline with branches created using `Map`?
??x
When a URL matches the pattern specified in the `Map` method, it follows the corresponding branch of middleware components defined within that map. The final component in the branch may or may not invoke the next delegate, which can result in different parts of the pipeline being executed based on the request path.

Example in C#:
```csharp
app.Map("/branch", branch => {
    branch.UseMiddleware<Platform.QueryStringMiddleWare>();
    branch.Use(async (HttpContext context, Func<Task> next) => 
    { 
        await context.Response.WriteAsync("Branch Middleware"); 
    });
});
```
In this example, requests to `/branch` will pass through `QueryStringMiddleWare` and an additional middleware that writes "Branch Middleware" to the response. Requests to other paths follow the main pipeline.

x??

---

---
#### Using `MapWhen` Middleware
Background context: The `MapWhen` middleware method allows you to branch the request pipeline based on certain conditions. This is useful for routing requests differently depending on specific query parameters or other factors.

:p What does the `app.MapWhen` method do, and how is it used in the example provided?
??x
The `app.MapWhen` method branches the request pipeline when a specified predicate function returns true. In the example, the middleware checks if the query string contains a key named "branch". If it does, certain middleware components are added to handle those requests.

Example code:
```csharp
app.MapWhen(context => context.Request.Query.Keys.Contains("branch"), 
    branch => {
        // Add middleware components here...
});
```
x??

---
#### Creating Terminal Middleware with `Run` Method
Background context: Terminal middleware is a type of middleware that does not forward requests to other components and always marks the end of the request pipeline. The `Run` method in ASP.NET Core simplifies creating terminal middleware by directly writing content to the response without calling the next middleware.

:p How does the `Run` method facilitate the creation of terminal middleware?
??x
The `Run` method is a convenience feature that allows you to write terminal middleware by immediately responding to the request and ending the pipeline. When using `Run`, no further middleware processing occurs for the request. This is demonstrated in Listing 12.12 where a simple message "Branch Middleware" is written directly to the response.

Example code:
```csharp
app.Use(async (context, next) => {
    await context.Response.WriteAsync("Branch Middleware");
});
```
This snippet uses `Use` with an async lambda that writes the string "Branch Middleware" to the response and then ends processing by not calling `next`.

x??

---
#### Applying Terminal Middleware Using `Run` Method
Background context: In ASP.NET Core, terminal middleware components end the request pipeline without forwarding requests. The `Run` method is used in these cases to ensure that no further middleware processes are called.

:p How does the `Run` method differ from using `Use` for creating a middleware component?
??x
The `Run` method is specifically designed for terminal middleware, meaning it ends the processing of the request pipeline and does not invoke any subsequent middleware. In contrast, `Use` can be used to add both regular and terminal middleware; if you want to ensure no further middleware runs after your current handler, use `Run`.

Example code:
```csharp
app.Use(async (context) => {
    await context.Response.WriteAsync("Branch Middleware");
});
```
This uses `Use` without calling the next function, making it a terminal component. However, if you want to make it explicit that this is a terminal handler, use `Run` instead.

x??

---
#### Adding Terminal Support in Class-based Middleware
Background context: Class-based components can be designed to work as both regular and terminal middleware by handling the `nextDelegate` parameter appropriately. When this parameter is provided (non-null), the component behaves like regular middleware; otherwise, it acts as a terminal component.

:p How does a class-based component handle the `nextDelegate` parameter to determine whether it should act as terminal or regular middleware?
??x
A class-based component can be designed to accept an optional `nextDelegate` parameter. If this parameter is null (indicating no further middleware), the component acts as a terminal component by directly responding to the request without calling the next function. Otherwise, it behaves like regular middleware by invoking the `nextDelegate`.

Example code:
```csharp
public class QueryStringMiddleWare {
    private RequestDelegate? next;
    
    public QueryStringMiddleWare() { }
    public QueryStringMiddleWare(RequestDelegate nextDelegate) {
        next = nextDelegate;
    }

    public async Task Invoke(HttpContext context) {
        if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
            if (context.Response.HasStarted) {
                context.Response.ContentType = "text/plain";
            }
            await context.Response.WriteAsync("Class Middleware");
        }
        
        if (next != null) {
            await next(context);
        }
    }
}
```
This class has two constructors and an `Invoke` method. The first constructor initializes the component without setting up a next delegate, making it suitable for terminal use cases. The second constructor sets the `nextDelegate`, allowing it to be used as regular middleware.

x??

---
#### Adding Terminal Support in Class-based Middleware (Different Description)
Background context: To ensure that class-based middleware can operate both as terminal and regular components, the component's logic must properly handle the scenario where no further processing should occur. This is achieved by checking if `next` is null before invoking it.

:p How does a class-based component check for terminal behavior in its `Invoke` method?
??x
In a class-based middleware component, the `Invoke` method checks whether the `nextDelegate` (referred to as `next`) is null. If `next` is null, it means that no further processing should occur, and the request pipeline ends here. Otherwise, if `next` is not null, the next middleware in the pipeline will be called.

Example code:
```csharp
public class QueryStringMiddleWare {
    private RequestDelegate? next;
    
    public async Task Invoke(HttpContext context) {
        // Conditional logic to check if this should act as terminal or regular middleware
        if (next != null) {
            await next(context);
        }
    }
}
```
This snippet shows that the `Invoke` method checks for a non-null `next` before proceeding to call it. If `next` is null, the request processing stops here.

x??

---

