# Flashcards: Pro-ASPNET-Core-7_processed (Part 95)

**Starting Chapter:** 15.3.2 Logging messages with attributes

---

#### Creating a Logger in Program.cs
Background context: This section demonstrates how to create and use a logger for logging messages during the startup of an ASP.NET Core application. The `ILogger` interface is used to log various debug information.

:p How do you create a logger using `ILoggerFactory` in the `Program.cs` file?
??x
To create a logger, you first need to retrieve the `ILoggerFactory` from the DI container and then use it to create an instance of `ILogger` for the specified category. Here's how it is done:

```csharp
var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("Pipeline");
```

This line retrieves the factory service and creates a logger named "Pipeline". The logger can now be used to log messages.

??x
The answer with detailed explanations.
To create a logger, you first need to retrieve the `ILoggerFactory` from the DI (Dependency Injection) container of your application. This is done using:

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("Pipeline");
```

The above code snippet shows how to retrieve the `ILoggerFactory` from the services collection and then create a logger named "Pipeline". This logger can be used throughout your application to log debug information. Here's an example of logging:

```csharp
logger.LogDebug("Pipeline configuration starting");
```

You can also see this in action when you run the application, where you will get messages like:
```
debug: Pipeline[0]       Pipeline configuration starting
```

Similarly, for the end of the setup process, you would log:

```csharp
logger.LogDebug("Pipeline configuration complete");
```

This helps to trace the flow and configuration status during startup.
x??

---

#### Logging Messages with Attributes
Background context: This section introduces using the `LoggerMessage` attribute in logging messages within a method. The attribute allows for more structured and easier-to-maintain logging compared to manually calling `LogDebug`, etc.

:p How can you use the `LoggerMessage` attribute to log messages?
??x
You can use the `LoggerMessage` attribute to generate structured logs by defining partial methods in a class. This approach is described in detail in this section, and it offers better performance than manual logging calls.

??x
The answer with detailed explanations.
To use the `LoggerMessage` attribute for logging, you define partial methods within a class. These methods are then automatically generated at compile time to include the necessary logging logic. Here's an example from the text:

```csharp
[LoggerMessage(0, LogLevel.Debug, "Starting response for {path}")]
public static partial void StartingResponse(ILogger logger, string path);
```

This attribute is applied to a method `StartingResponse` that takes an `ILogger` instance and a `string` parameter. When the application compiles, it generates the necessary implementation for this method.

Here's how you can use it in your code:

```csharp
public static async Task Endpoint(HttpContext context, ILogger<Population> logger)
{
    // ...
    StartingResponse(logger, context.Request.Path);
    // ...
}
```

When `StartingResponse` is called with an `ILogger` and a path string, the attribute ensures that a debug log message is generated for you:

```plaintext
dbug: Platform.Population[0]       Starting response for /population
```

This approach simplifies logging by abstracting away the details of generating the logs.

??x
The answer with detailed explanations.
To use the `LoggerMessage` attribute, you define a partial method in your class. This method is marked with the `LoggerMessage` attribute and takes an `ILogger` instance as one of its parameters:

```csharp
[LoggerMessage(0, LogLevel.Debug, "Starting response for {path}")]
public static partial void StartingResponse(ILogger logger, string path);
```

The first two parameters are important: `0` is a unique identifier that helps distinguish this log message from others, and `LogLevel.Debug` specifies the level of the log. The third parameter `"Starting response for {path}"` is a format string used to log the message.

In your class, you can call this method:

```csharp
public static async Task Endpoint(HttpContext context, ILogger<Population> logger)
{
    // ...
    StartingResponse(logger, context.Request.Path);
    // ...
}
```

When `StartingResponse` is called with an `ILogger` and a path string, the attribute ensures that a debug log message is generated for you. This helps maintain cleaner code while ensuring structured logging.

??x
The answer with detailed explanations.
To use the `LoggerMessage` attribute effectively, ensure your class is defined as a partial class. Here's a complete example of how it works:

```csharp
namespace Platform {
    public partial class Population {
        public static async Task Endpoint(HttpContext context, ILogger<Population> logger) 
        {
            StartingResponse(logger, context.Request.Path);
            
            string city = context.Request.RouteValues["city"] as string ?? "london";
            int? pop = null;
            switch (city.ToLower()) {
                case "london":
                    pop = 8_136_000;
                    break;
                case "paris":
                    pop = 2_141_000;
                    break;
                case "monaco":
                    pop = 39_000;
                    break;
            }
            
            if (pop.HasValue) {
                await context.Response.WriteAsync($"City: {city}, Population: {pop}");
            } else {
                context.Response.StatusCode = StatusCodes.Status404NotFound;
            }

            logger.LogDebug("Finished processing for {path}", context.Request.Path);
        }

        [LoggerMessage(0, LogLevel.Debug, "Starting response for {path}")]
        public static partial void StartingResponse(ILogger logger, string path);
    }
}
```

When you call `StartingResponse(logger, context.Request.Path);`, the attribute ensures that a debug log message is generated with the provided format and parameters. This approach provides better performance compared to manually calling logging methods.

---

#### Configuring Minimum Logging Levels
Background context: In ASP.NET Core, the `appsettings.json` and `appsettings.Development.json` files are used to configure logging. The `Logging:LogLevel` section sets the minimum level for logging messages. Log messages below this level are discarded.

:p How does one configure the minimum log level in `appsettings.json`?
??x
You configure the minimum log level by setting keys under the `Logging:LogLevel` object in the `appsettings.json` file. For example, to set the default logging level to Information and the Microsoft category to Warning, you would add or update this section:

```json
{
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

#### Debugging with Specific Log Levels
Background context: During development, it is common to increase the log detail by setting specific categories to higher logging levels like `Debug`. This allows developers to focus on specific parts of their application.

:p How can you configure more detailed logging for a specific namespace in `appsettings.Development.json`?
??x
To configure more detailed logging for a specific namespace, modify the `Logging:LogLevel` section in the `appsettings.Development.json` file. For example, to set the Microsoft.AspNetCore and Microsoft.AspNetCore.Routing categories to Debug:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Warning",
      "Microsoft.AspNetCore.Routing": "Debug"
    }
  }
}
```
x??

---

#### Filtering Log Messages by Category
Background context: Log messages can be filtered by their category. The category is set using the generic type argument or a string. Categories like `Platform.Population` allow for direct filtering, while categories not explicitly configured fall back to the Default entry.

:p How do you filter log messages generated by the Population class?
??x
To filter log messages generated by the Population class, you can use the category `Platform.Population`. This means that such log messages will be matched directly if the category is specified in the `Logging:LogLevel` section. For example:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Platform.Population": "Debug"
    }
  }
}
```
x??

---

#### Reducing Log Detail by Being More Specific
Background context: You can reduce the log detail by being more specific about which namespace requires messages. This allows for a finer granularity of control over what logs are displayed.

:p How does specifying a more specific namespace affect logging levels?
??x
Specifying a more specific namespace in the `Logging:LogLevel` section will override the default settings and apply new logging levels only to that category. For instance, setting the Microsoft.AspNetCore.Routing category to Debug while keeping other categories at their current level:

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Warning",
      "Microsoft.AspNetCore.Routing": "Debug"
    }
  }
}
```
x??

---

---
#### Adding HTTP Logging Middleware
Background context: This section explains how to add logging middleware for HTTP requests and responses in an ASP.NET Core application. The `UseHttpLogging` method adds a middleware component that generates log messages describing the HTTP requests and responses.

:p How do you add HTTP logging middleware to your ASP.NET Core application?
??x
To add HTTP logging middleware, you use the `app.UseHttpLogging();` method within the `Program.cs` file. This method configures the `Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware` to generate logs for each request and response.

Example in `Program.cs`:
```csharp
using Platform;
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.UseHttpLogging(); // Add this line to enable HTTP logging
```
x??
---
#### Configuring Logging Levels
Background context: This section describes how to configure the logging levels in `appsettings.Development.json` to ensure that the logs for specific categories are displayed.

:p How do you configure the log levels for HTTP logging middleware in ASP.NET Core?
??x
You configure the log levels by modifying the `appsettings.Development.json` file. Specifically, you set the appropriate severity level for the `Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware`.

Example configuration:
```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Warning",
      "Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware": "Information"
    }
  }
}
```
x??
---
#### Logging HTTP Request and Response Details
Background context: This section provides an example of logging detailed information about the HTTP request and response.

:p What are the log messages generated by `UseHttpLogging` in an ASP.NET Core application?
??x
The `UseHttpLogging` middleware generates detailed logs for each HTTP request and response. The logs include fields like the protocol, method, path, status code, etc.

Example log messages:
```
info: Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware[1]
Request:
Protocol: HTTP/1.1
Method: GET
Scheme: http
PathBase:
Path: /population
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,...
Connection: keep-alive
Host: localhost:5000
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) ...
```

```csharp
info: Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware[2]
Response:
StatusCode: 200
Date: [Redacted]
Server: [Redacted]
Transfer-Encoding: chunked
```
x??
---
#### Configuring HTTP Logging Fields
Background context: This section explains how to customize the fields that are logged for HTTP requests and responses.

:p How do you configure which fields are logged by `UseHttpLogging` in ASP.NET Core?
??x
You can configure the fields to log by using the `AddHttpLogging` method in your `Program.cs`. You specify the fields you want to include or exclude.

Example configuration:
```csharp
using Platform;
using Microsoft.AspNetCore.HttpLogging;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddHttpLogging(opts => {
    opts.LoggingFields = HttpLoggingFields.RequestMethod 
                         | HttpLoggingFields.RequestPath 
                         | HttpLoggingFields.ResponseStatusCode;
});
```
x??
---

