# Flashcards: Pro-ASPNET-Core-7_processed (Part 35)

**Starting Chapter:** 12.3 Understanding the ASP.NET Core project

---

---
#### Middleware Components in ASP.NET Core
Middleware components are a critical part of the ASP.NET Core pipeline, designed to handle specific tasks and pass requests through them. They can generate responses or provide supporting features such as formatting data types or reading/writing cookies. If no middleware component generates a response, ASP.NET Core returns an HTTP 404 Not Found status code.

:p What are middleware components in the context of ASP.NET Core?
??x
Middleware components in ASP.NET Core handle specific tasks within the request-response pipeline and can generate responses or provide supporting features like data formatting. If no middleware component generates a response, the application defaults to returning an HTTP 404 Not Found status code.
x??

---
#### Services in ASP.NET Core
Services are objects that provide functionality within a web application. Any class can be used as a service, and services managed by ASP.NET Core support dependency injection, allowing for easy access from anywhere in the application, including middleware components.

:p What are services in an ASP.NET Core application?
??x
Services in an ASP.NET Core application are objects that provide specific functionality within the web app. They can be any class and are managed by the platform to support dependency injection, enabling easy access across different parts of the application, such as middleware components.
x??

---
#### Dependency Injection in Services
Dependency injection is a design pattern used to inject dependencies into services, making them more flexible and testable. In ASP.NET Core, this enables seamless integration and sharing of services among various components.

:p What is dependency injection?
??x
Dependency injection (DI) is a design pattern that involves passing dependencies to classes rather than having them created within the class itself. This makes the code more modular and easier to test. In ASP.NET Core, DI allows for easy access to shared services across different parts of the application, such as middleware components.
x??

---
#### Files in an Example Project
The web template provides a basic set of files to start an ASP.NET Core project with some initial configuration. These files are essential for running and configuring the application.

:p What files are included in the example project?
??x
The example project includes several key files that configure and run the ASP.NET Core application, such as `appsettings.json`, `Program.cs`, and configuration JSON files like `appsettings.Development.json`. Other important files include build artifacts (`bin`), global settings (`global.json`), and project metadata (`Platform.csproj`).

The full list of files in the example project is:
- `appsettings.json`: Configures the application.
- `appsettings.Development.json`: Contains development-specific configuration.
- `bin`: Compiled application files (hidden by Visual Studio).
- `global.json`: Selects specific versions of .NET Core SDK.
- `Properties/launchSettings.json`: Starts the application with configurations.
- `obj`: Intermediate output from the compiler (hidden by Visual Studio).
- `Platform.csproj`: Describes the project to .NET Core tools.
- `Platform.sln`: Organizes projects.

Example of a file in the example project:
```json
{
  "ConnectionStrings": {
    "DefaultConnection": "Server=(localdb)\\mssqllocaldb;Database=aspnet-CoreWebApp-17460a3e-2d85-4f97-8b3c-efc1d82a96f1;Trusted_Connection=True;MultipleActiveResultSets=true"
  },
  "Logging": {
    "LogLevel": {
      "Default": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```
x??

---
#### Understanding the Entry Point
`Program.cs` is the entry point of the ASP.NET Core platform. It is used to configure and run the application, setting up services and middleware components.

:p What does `Program.cs` do in an ASP.NET Core project?
??x
`Program.cs` serves as the entry point for the ASP.NET Core platform. It configures the web host, sets up services and middleware components, and starts the application. The following code snippet shows how this is typically done:
```csharp
public class Program
{
    public static void Main(string[] args)
    {
        CreateHostBuilder(args).Build().Run();
    }

    public static IHostBuilder CreateHostBuilder(string[] args) =>
        Host.CreateDefaultBuilder(args)
            .ConfigureWebHostDefaults(webBuilder =>
            {
                webBuilder.UseStartup<Startup>();
            });
}
```
In this example, the `CreateHostBuilder` method configures and builds the host, which then runs the application using the `Startup` class.
x??

---

#### ASP.NET Core Entry Point and Program.cs File
Background context explaining that the `Program.cs` file contains the entry point for an ASP.NET Core application. It sets up services, defines routes, and starts the HTTP server. The `WebApplication.CreateBuilder(args)` method initializes the setup of the platform, while the `app.MapGet("/", () => "Hello World.");` function maps a GET request to the root URL.
:p What is the purpose of the `Program.cs` file in an ASP.NET Core application?
??x
The `Program.cs` file serves as the entry point for an ASP.NET Core application. It initializes the platform setup, registers services, defines routes, and starts the HTTP server using Kestrel.

Code example:
```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", () => "Hello World.");
app.Run();
```
x??

---

#### WebApplicationBuilder and Build Method
Background context explaining that `WebApplicationBuilder` is used to configure the application, while the `Build()` method finalizes this configuration. The result of `Build()` is a `WebApplication` object.
:p What does the `WebApplicationBuilder` class do in ASP.NET Core?
??x
The `WebApplicationBuilder` class is responsible for configuring an ASP.NET Core application by registering services and setting up middleware. It provides methods to add services, configure settings, and other setup tasks.

Code example:
```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
```
x??

---

#### MapGet Method and Middleware Setup
Background context explaining that `MapGet` is an extension method for the `IEndpointRouteBuilder` interface. It defines a route for handling HTTP GET requests.
:p How does the `MapGet` method work in ASP.NET Core?
??x
The `MapGet` method sets up a middleware component to handle HTTP GET requests with a specified URL path. In this example, it maps a GET request to the root URL (`/`) and responds with "Hello World."

Code example:
```csharp
app.MapGet("/", () => "Hello World.");
```
x??

---

#### Kestrel as the HTTP Server
Background context explaining that Kestrel is an HTTP server used in ASP.NET Core applications. It receives HTTP requests from clients.
:p What is Kestrel and its role in ASP.NET Core?
??x
Kestrel is the HTTP server used in ASP.NET Core applications to receive HTTP requests from clients. It is a fast, lightweight, cross-platform web server that serves as the entry point for incoming HTTP traffic.

Code example:
```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.Run(); // Starts Kestrel to listen to HTTP requests.
```
x??

---

#### Running the Application and Testing with PowerShell
Background context explaining how running an ASP.NET Core application using `Run` method in `Program.cs`. Also, instructions on how to test it via PowerShell command.

:p How can you run and test an ASP.NET Core application from the command line?
??x
To run an ASP.NET Core application, call the `Run` method in `Program.cs`, which starts Kestrel to listen for HTTP requests. To test it, open a new PowerShell prompt and use the following command:
```powershell
Invoke-WebRequest http://localhost:5000 | Select-Object -ExpandProperty RawContent
```
This command sends an HTTP GET request to `http://localhost:5000` and displays the raw response content.

Code example for testing in PowerShell is given above.
x??

---

#### Adding a Package to a Project Using `dotnet add package`
Background context: In most .NET projects, adding dependencies is often done through command-line tools or integrated development environment (IDE) interfaces. This method allows developers to easily include third-party libraries without modifying project files directly.

To add a package via the command line, you can use the following steps:
1. Open a new PowerShell command prompt.
2. Navigate to the root directory of your .NET project that contains the `.csproj` file.
3. Use the `dotnet add package` command followed by the name and version number of the package.

:p How do you add a package using the `dotnet add package` command in a .NET project?
??x
The `dotnet add package` command is used to install a NuGet package into your .NET project. This method updates the `.csproj` file to include a new `<PackageReference>` entry for the specified package and version.

Example command:
```sh
dotnet add package Swashbuckle.AspNetCore --version 6.4.0
```

This command adds the `Swashbuckle.AspNetCore` package with version `6.4.0` to your project. The updated `.csproj` file will look like this:

```xml
<Project Sdk="Microsoft.NET.Sdk.Web">
    <PropertyGroup>
        <TargetFramework>net7.0</TargetFramework>
        <Nullable>enable</Nullable>
        <ImplicitUsings>enable</ImplicitUsings>
    </PropertyGroup>
    <ItemGroup>
        <PackageReference Include="Swashbuckle.AspNetCore" Version="6.4.0" />
    </ItemGroup>
</Project>
```

x??

---

#### Creating Custom Middleware in ASP.NET Core
Background context: ASP.NET Core provides a flexible middleware pipeline that allows developers to create and chain custom components to handle HTTP requests and responses. This is useful for implementing cross-cutting concerns like logging, authentication, or custom behavior.

:p How do you create custom middleware in an ASP.NET Core application?
??x
To create custom middleware in an ASP.NET Core application, you use the `app.Use` method within the `Program.cs` file. The `Use` method takes a lambda function that represents your middleware logic and passes each request through it.

Example code:
```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.Use(async (context, next) => {
    if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
        context.Response.ContentType = "text/plain";
        await context.Response.WriteAsync("Custom Middleware");
    }

    await next();
});

app.MapGet("/", () => "Hello World.");
app.Run();
```

In this example, the middleware checks if the request method is a GET and has a query parameter `custom` set to `true`. If so, it sets the response content type and writes custom text. Otherwise, it passes the request to the next middleware component in the pipeline.

x??

---

#### Understanding the HttpContext Class
Background context: The `HttpContext` class in ASP.NET Core is a powerful object that provides information about the current HTTP request and response. It contains various properties and methods that allow you to interact with the request and response objects, as well as other useful features like sessions and user details.

:p What are some important members of the HttpContext class?
??x
The `HttpContext` class in ASP.NET Core is part of the `Microsoft.AspNetCore.Http` namespace and provides several useful members for handling HTTP requests and responses. Here are a few key properties:

- **Connection**: Returns a `ConnectionInfo` object that contains information about the underlying network connection.
- **Request**: Returns an `HttpRequest` object representing the current request.
- **Response**: Returns an `HttpResponse` object used to create the response to the HTTP request.
- **Session**: Returns session data associated with the request.
- **User**: Provides details of the user making the request.

Example code:
```csharp
public class ExampleMiddleware {
    public async Task InvokeAsync(HttpContext context) {
        // Accessing the Request property
        if (context.Request.Method == "GET" && context.Request.Query["custom"] == "true") {
            context.Response.ContentType = "text/plain";
            await context.Response.WriteAsync("Custom Middleware");
        }

        // Passing to the next middleware or endpoint
        await next(context);
    }
}
```

x??

---

#### ASP.NET Core HttpRequest Class Members
Background context: The `HttpRequest` class is a crucial part of handling HTTP requests in ASP.NET Core. It encapsulates all the data and methods needed to process an incoming request, making it easier for middleware and endpoints to focus on their specific tasks.

:p List some key members of the `HttpRequest` class.
??x
The `HttpRequest` class provides several useful properties that make handling HTTP requests more manageable:

- **Body**: Returns a stream used to read the request body.
- **ContentLength**: Returns the length of the content in the request.
- **ContentType**: Returns the Content-Type header value.
- **Cookies**: Returns cookies from the request.
- **Form**: Represents form data, if available.
- **Headers**: Contains all headers associated with the request.
- **IsHttps**: Indicates whether the request was made using HTTPS.
- **Method**: Gets the HTTP verb used for the request (e.g., GET, POST).
- **Path**: Returns the path section of the request URL.
- **Query**: Provides key-value pairs from the query string.

For example, you can check if a request is made via HTTP or HTTPS:

```csharp
if (context.Request.IsHttps)
{
    // Do something when it's an HTTPS request
}
```

Or retrieve specific query parameters like this:

```csharp
var customParam = context.Request.Query["custom"];
if (customParam == "true")
{
    // Handle the GET request with the 'custom' parameter set to true
}
```
x??

---

#### ASP.NET Core HttpResponse Class Members
Background context: The `HttpResponse` class is essential for constructing and sending responses back to the client. It provides various methods and properties to control headers, content type, status codes, and more.

:p List some key members of the `HttpResponse` class.
??x
The `HttpResponse` class offers a variety of useful properties and methods to handle HTTP responses:

- **ContentLength**: Sets the Content-Length header.
- **ContentType**: Sets the Content-Type header.
- **Cookies**: Allows setting cookies with the response.
- **HasStarted**: Indicates whether ASP.NET Core has started sending response headers.
- **Headers**: Sets or retrieves response headers.
- **StatusCode**: Sets the status code for the response.
- **WriteAsync(data)**: Asynchronously writes a string to the response body.
- **Redirect(url)**: Sends a redirection response.

For instance, setting the content type and writing a simple text response:

```csharp
context.Response.ContentType = "text/plain";
await context.Response.WriteAsync("Custom Middleware");
```

Or setting up a redirect:

```csharp
context.Response.Redirect("/home/index");
```
x??

---

#### Custom Middleware in ASP.NET Core
Background context: Custom middleware is a powerful feature of ASP.NET Core that allows developers to insert custom logic into the request pipeline. This can be used for various purposes, such as logging, authentication, or content transformation.

:p What does the `next` parameter in custom middleware represent?
??x
The `next` parameter in custom middleware represents the function that tells ASP.NET Core to pass the current HTTP context (including the request and response) to the next component in the pipeline. This allows you to chain multiple middlewares together, ensuring each one can process the request before it reaches its final destination.

For example:

```csharp
public async Task InvokeAsync(HttpContext context)
{
    if (context.Request.Method == HttpMethods.Get &&
        context.Request.Query["custom"] == "true")
    {
        context.Response.ContentType = "text/plain";
        await context.Response.WriteAsync("Custom Middleware");
    }
    
    // Pass the request to the next middleware in the pipeline
    await next(context);
}
```

Here, `await next(context);` ensures that the request is processed by any subsequent middlewares or endpoints.

x??

---

#### HttpContext Object in Custom Middleware
Background context: The `HttpContext` object is a central object that holds all the information about both the incoming HTTP request and the outgoing response. It's used directly when writing custom middleware but is less commonly needed with higher-level frameworks like MVC or Razor Pages.

:p How does the `HttpContext` object manage requests in custom middleware?
??x
The `HttpContext` object manages requests by providing access to various properties of the incoming request and methods for sending responses back to the client. In custom middleware, it's used to inspect and modify the HTTP context as needed.

For example:

```csharp
public async Task InvokeAsync(HttpContext context)
{
    if (context.Request.Method == HttpMethods.Get &&
        context.Request.Query["custom"] == "true")
    {
        // Set content type and write a simple response
        context.Response.ContentType = "text/plain";
        await context.Response.WriteAsync("Custom Middleware");
        
        // Pass the request to the next middleware in the pipeline
        await next(context);
    }
}
```

In this example, `HttpContext` is used to check if the request matches certain criteria and then modify the response accordingly.

x??

---

---
#### Custom Middleware Explanation
Custom middleware can be defined using lambda functions or classes. Lambda functions are convenient but may lead to long and complex statements, making it hard to reuse across projects. Classes provide a way to keep middleware code organized outside of `Program.cs`.

Lambda function example:
```csharp
app.Use(async (context, next) => {
    if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
        context.Response.ContentType = "text/plain";
        await context.Response.WriteAsync("Custom Middleware ");
    }
    await next();
});
```

Class-based middleware example:
```csharp
public class QueryStringMiddleWare {
    private RequestDelegate next;
    public QueryStringMiddleWare(RequestDelegate nextDelegate) {
        next = nextDelegate;
    }

    public async Task Invoke(HttpContext context) {
        if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
            if (!context.Response.HasStarted) { // Fix: Change .context to context
                context.Response.ContentType = "text/plain";
            }
            await context.Response.WriteAsync("Class Middleware ");
        }
        await next(context);
    }
}
```

:p How does class-based middleware differ from lambda function middleware in ASP.NET Core?
??x
Class-based middleware differs by receiving a `RequestDelegate` object as a constructor parameter, allowing requests to be forwarded asynchronously. The `Invoke` method processes incoming requests and can conditionally execute custom logic before passing the request to the next middleware component.

Code example:
```csharp
public class QueryStringMiddleWare {
    private RequestDelegate next;
    public QueryStringMiddleWare(RequestDelegate nextDelegate) {
        next = nextDelegate;
    }

    public async Task Invoke(HttpContext context) {
        if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
            if (!context.Response.HasStarted) { // Fix: Change .context to context
                context.Response.ContentType = "text/plain";
            }
            await context.Response.WriteAsync("Class Middleware ");
        }
        await next(context);
    }
}
```
x??

---
#### Adding Class-Based Middleware
To add class-based middleware, use the `UseMiddleware<T>` method in `Program.cs`, where `T` is the type of your middleware class. This method allows you to integrate custom logic outside the main entry point of the application.

:p How do you add a class-based middleware component to the ASP.NET Core pipeline?
??x
You add a class-based middleware component by using the `UseMiddleware<T>` method in `Program.cs`, where `T` is the type of your middleware class. This integration keeps the custom logic separate from the main entry point, making it reusable across different projects.

Example:
```csharp
app.UseMiddleware<Platform.QueryStringMiddleWare>();
```

x??

---
#### Thread-Safety Considerations for Middleware
Since a single middleware object handles all requests in ASP.NET Core, any code within its `Invoke` method must be thread-safe. This is crucial because multiple requests can be processed concurrently.

:p Why is the `Invoke` method of class-based middleware important?
??x The `Invoke` method of class-based middleware is critical because it is called by ASP.NET Core for each incoming request. Since a single middleware instance handles all requests, the code within this method must ensure thread safety to avoid race conditions and other concurrency issues.

Example:
```csharp
public async Task Invoke(HttpContext context) {
    if (context.Request.Method == HttpMethods.Get && context.Request.Query["custom"] == "true") {
        if (!context.Response.HasStarted) { // Fix: Change .context to context
            context.Response.ContentType = "text/plain";
        }
        await context.Response.WriteAsync("Class Middleware ");
    }
    await next(context);
}
```

x??

---

