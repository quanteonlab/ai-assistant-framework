# Flashcards: Pro-ASPNET-Core-7_processed (Part 47)

**Starting Chapter:** 15.3.2 Logging messages with attributes

---

#### Creating a Logger in ASP.NET Core
Background context: In this section, we explore how to use logging services provided by ASP.NET Core. The `ILogger` and `ILoggerFactory` are used to log messages during the application's lifecycle.

:p How do you create a logger for a specific category in an ASP.NET Core application?
??x
To create a logger for a specific category, you can use the `CreateLogger` method from `ILoggerFactory`. This is typically done within the main entry point of your application, such as the `Program.cs` file.

```csharp
using Microsoft.Extensions.Logging;
// ...

var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("Pipeline");
logger.LogDebug("Pipeline configuration starting");
app.MapGet("population/{city?}", Population.Endpoint);
logger.LogDebug("Pipeline configuration complete");
app.Run();
```

x??

---

#### Logging Messages with Attributes
Background context: The `LoggerMessage` attribute is an alternative way to generate log messages in ASP.NET Core. It provides better performance compared to other logging methods because the implementation is generated during compilation.

:p How does the `LoggerMessage` attribute work, and what are its benefits?
??x
The `LoggerMessage` attribute works by applying it to partial methods within a partial class. When the application is compiled, it generates the necessary implementation for these methods, which results in more efficient logging compared to other techniques.

```csharp
namespace Platform {
    public partial class Population {
        [LoggerMessage(0, LogLevel.Debug, "Starting response for {path}")]
        public static partial void StartingResponse(ILogger logger, string path);

        public static async Task Endpoint(HttpContext context, ILogger<Population> logger) {
            // Logger messages can be generated using the attribute
            StartingResponse(logger, context.Request.Path);
            
            // Code to process request and log completion
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
    }
}
```

The main benefits are:
1. Improved performance due to generated implementation.
2. Easier maintenance and understanding of logging code.

x??

---
#### Logging with `CreateLogger` Method
Background context: The `CreateLogger` method is a straightforward way to obtain an `ILogger` instance for specific parts of your application, allowing you to log messages at different levels such as Debug or Info.

:p What are the steps to use `CreateLogger` in ASP.NET Core?
??x
To use `CreateLogger`, follow these steps:
1. Obtain the `ILoggerFactory` from the service collection.
2. Use the factory's `CreateLogger` method, providing a category name for your logger.
3. Call methods on the created `ILogger` instance to log messages.

```csharp
var logger = app.Services.GetRequiredService<ILoggerFactory>().CreateLogger("Pipeline");
logger.LogDebug("Pipeline configuration starting");
```

x??

---

#### Using `LoggerMessage` Attribute in ASP.NET Core
Background context: The `LoggerMessage` attribute is used within partial classes and methods to generate log messages more efficiently. It leverages the Roslyn compiler for better performance.

:p What are the steps to use the `LoggerMessage` attribute?
??x
To use the `LoggerMessage` attribute, follow these steps:
1. Define a method as a partial method in a partial class.
2. Apply the `LoggerMessage` attribute to this method, specifying the message ID, log level, and message format.
3. Call the generated method from your application logic.

```csharp
namespace Platform {
    public partial class Population {
        [LoggerMessage(0, LogLevel.Debug, "Starting response for {path}")]
        public static partial void StartingResponse(ILogger logger, string path);
        
        public static async Task Endpoint(HttpContext context, ILogger<Population> logger) {
            StartingResponse(logger, context.Request.Path); // Generated method call
            // ... process request and log completion ...
            logger.LogDebug("Finished processing for {path}", context.Request.Path);
        }
    }
}
```

x??

---

---
#### Configuring Minimum Logging Levels in ASP.NET Core
Background context explaining how logging is configured using JSON files like `appsettings.json` and `appsettings.Development.json`. These files are merged to set up the logging service for an ASP.NET Core application. The `Logging:LogLevel` section of these JSON files determines which log messages get recorded based on their level (e.g., Debug, Information, Warning).

The default configuration in `appsettings.json` sets the minimum log levels for different categories such as `Default`, `Microsoft.AspNetCore`. This means that any log message below this level is discarded.

:p How does ASP.NET Core determine whether to log a message?
??x
ASP.NET Core determines whether to log a message based on the minimum log level set in the `Logging:LogLevel` section of the configuration files. For example, if the default level is set to "Information", then only messages with levels "Information" and above will be logged.

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
#### Setting Custom Logging Levels in `appsettings.Development.json`
Background context explaining that during development, more detailed logging is often required. The `appsettings.Development.json` file allows you to adjust the log levels for specific categories or globally.

:p How does setting custom logging levels in `appsettings.Development.json` affect the application?
??x
Setting custom logging levels in `appsettings.Development.json` increases the detail of log messages, providing more insight into the application's state during development. For example, you can set the default level to "Debug" and specific categories like "System" or "Microsoft.AspNetCore" to "Information".

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Debug"
    }
  }
}
```
x??

---
#### Filtering Log Messages by Category
Background context explaining how log messages are filtered based on their category. You can specify different minimum logging levels for various categories in the `Logging:LogLevel` section.

:p How does specifying a logging level for a specific category affect the application?
??x
Specifying a logging level for a specific category ensures that only log messages from that category with or above that level are logged. For example, setting the "Microsoft.AspNetCore" category to "Warning" will only log warning and error-level messages from this namespace.

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "Information",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```
x??

---
#### Disabling Logging for a Category
Background context explaining that you can disable logging for a specific category by setting its log level to "None". This is useful when you want to reduce the amount of logging without affecting other categories.

:p How does disabling logging for a category affect the application?
??x
Disabling logging for a category sets its log level to "None", which means that no messages from that category will be logged. For example, setting the "System" category to "None" will disable all system-related logs.

```json
{
  "Logging": {
    "LogLevel": {
      "Default": "Debug",
      "System": "None",
      "Microsoft": "Information",
      "Microsoft.AspNetCore": "Warning"
    }
  }
}
```
x??

---
#### Adjusting Logging Levels for Specific Components
Background context explaining that you can adjust the logging levels to focus on specific parts of the application. This is useful during development or troubleshooting.

:p How does adjusting the logging level for a specific component affect the logs?
??x
Adjusting the logging level for a specific component increases the detail of log messages generated by that component. For example, setting the "Microsoft.AspNetCore.Routing" category to "Debug" will provide more detailed routing-related logs.

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
#### Adding Logging Middleware to ASP.NET Core Application
Background context: This concept involves integrating logging middleware into an ASP.NET Core application for debugging and monitoring purposes. The `HttpLoggingMiddleware` generates logs for HTTP requests and responses, which can be useful when troubleshooting routing issues or understanding how the application handles incoming requests.

:p How does adding `UseHttpLogging()` in the `Program.cs` file affect the application?
??x
Adding `app.UseHttpLogging();` to the request pipeline of an ASP.NET Core application integrates logging middleware that generates logs for HTTP requests and responses. These logs are categorized under "Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware" with an Information severity level, allowing you to trace the interaction between the browser and your application.

```csharp
using Platform;
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();
app.UseHttpLogging(); // Adds HTTP logging middleware.
```
x??
---

#### Configuring Logging in `appsettings.Development.json`
Background context: The configuration of log levels is specified in the `appsettings.Development.json` file. This helps control what types of logs are generated and at which severity level they should be recorded.

:p How does configuring logging settings in `appsettings.Development.json` affect the application?
??x
Configuring logging settings in `appsettings.Development.json` allows you to specify the log levels for different categories, such as default messages, system-related messages, or those generated by specific middleware components like HTTP logging. In this case, setting the log level for `Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware` to "Information" ensures that relevant logs are captured.

```json
{
   \"Logging\": {
     \"LogLevel\": { 
       \"Default\": \"Debug\", 
       \"System\": \"Information\", 
       \"Microsoft\": \"Information\", 
       \"Microsoft.AspNetCore\": \"Warning\", 
       \"Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware\":  "Information"
     } 
   }
}
```
x??
---

#### HTTP Logging Middleware in Action
Background context: The `UseHttpLogging()` method adds middleware that generates logs for every HTTP request and response. These logs provide detailed information about the requests, including headers, status codes, and more.

:p What does a typical log entry generated by `UseHttpLogging()` look like?
??x
A typical log entry generated by `UseHttpLogging()` includes details such as the HTTP method, path, query parameters, headers, and response status code. Here is an example of what one might look like:

```
info: Microsoft.AspNetCore.HttpLogging.HttpLoggingMiddleware[1]
Request:
Protocol: HTTP/1.1
Method: GET
Scheme: http
PathBase:
Path: /population
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8, application/signed-exchange;v=b3;q=0.9
Connection: keep-alive
Host: localhost:5000
User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/94.0.4606.71 Safari/537.36
Accept-Encoding: gzip, deflate, br
Accept-Language: en-GB,en-US;q=0.9,en;q=0.8
Cache-Control: [Redacted]
Cookie: [Redacted]
Upgrade-Insecure-Requests: [Redacted]
sec-ch-ua: [Redacted]
sec-ch-ua-mobile: [Redacted]
sec-ch-ua-platform: [Redacted]
Sec-Fetch-Site: [Redacted]
Sec-Fetch-Mode: [Redacted]
Sec-Fetch-User: [Redacted]
Sec-Fetch-Dest: [Redacted]

Response:
StatusCode: 200
Date: [Redacted]
Server: [Redacted]
Transfer-Encoding: chunked
```
x??
---

#### Customizing HTTP Logging Fields
Background context: The `AddHttpLogging` method allows for custom configuration of the fields included in the logs. By default, it includes various request and response details. However, you can customize which fields are logged to reduce log size or focus on specific aspects.

:p How can you configure which fields are logged by `UseHttpLogging()`?
??x
You can configure which fields are logged using the `AddHttpLogging` method in the `Program.cs` file. By setting the appropriate flags, you can control what information is included in the logs. For example, to include only the request method and path, as well as the response status code, you would use:

```csharp
builder.Services.AddHttpLogging(opts => {
    opts.LoggingFields = HttpLoggingFields.RequestMethod 
                         | HttpLoggingFields.RequestPath  
                         | HttpLoggingFields.ResponseStatusCode;
});
```
x??
---

