# Flashcards: Pro-ASPNET-Core-7_processed (Part 94)

**Starting Chapter:** 15.2.5 Understanding the launch settings file

---

#### Understanding the launchSettings.json File
The `launchSettings.json` file is located within the `Properties` folder of an ASP.NET Core project. It contains configuration settings used for starting the platform, such as TCP ports and environment variables. This file is crucial for setting up different profiles based on development or production environments.
:p What does the `launchSettings.json` file contain?
??x
The `launchSettings.json` file includes configurations like application URLs, environment-specific settings (e.g., `ASPNETCORE_ENVIRONMENT`), and other launch settings specific to IIS Express or the project itself. For example:

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
    "Platform": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": true,
      "applicationUrl": "http://localhost:5000",
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
x??

---

#### Environment Variables in Launch Settings
Environment variables like `ASPNETCORE_ENVIRONMENT` are defined within the `launchSettings.json` file and affect how the application is configured at runtime. The value of this variable determines which configuration files (e.g., `appsettings.Development.json`, `appsettings.Production.json`) are loaded.
:p How do environment variables in `launchSettings.json` influence an ASP.NET Core application?
??x
Environment variables like `ASPNETCORE_ENVIRONMENT` determine the runtime behavior and configuration of the application. For instance, setting `ASPNETCORE_ENVIRONMENT` to `Development` will load `appsettings.Development.json`, whereas setting it to `Production` will load `appsettings.Production.json`.

To change the environment in Visual Studio, go to `Debug > Launch Profiles`. In Visual Studio Code, edit the `launch.json` file:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": ".NET Core Launch (web)",
      "type": "coreclr",
      "request": "launch",
      "preLaunchTask": "build",
      "program": "${workspaceFolder}/bin/Debug/net7.0/Platform.dll",
      "args": [],
      "cwd": "${workspaceFolder}",
      "stopAtEntry": false,
      "serverReadyAction": {
        "action": "openExternally",
        "pattern": "\\bNow listening on:\\s+(https?://\\S+)"
      },
      "env": {
        "ASPNETCORE_ENVIRONMENT": "Production"
      },
      "sourceFileMap": {
        "/Views": "${workspaceFolder}/Views"
      }
    },
    {
      "name": ".NET Core Attach",
      "type": "coreclr",
      "request": "attach"
    }
  ]
}
```
x??

---

#### Displaying Configuration Settings via Middleware
In the `Program.cs` file, configuration settings can be displayed by using middleware components. The example provided maps a GET request to the `/config` URL to display both default and environment-specific configurations.
:p How does one display configuration settings in an ASP.NET Core application?
??x
To display configuration settings, you can use middleware like `LocationMiddleware`. In the `Program.cs` file:

```csharp
using Microsoft.Extensions.Configuration;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

var builder = WebApplication.CreateBuilder(args);
var servicesConfig = builder.Configuration;
builder.Services.Configure<MessageOptions>(servicesConfig.GetSection("Location"));
var app = builder.Build();
var pipelineConfig = app.Configuration;

app.UseMiddleware<LocationMiddleware>();
app.MapGet("/config", async (HttpContext context, IConfiguration config) => {
    string? defaultDebug = config["Logging:LogLevel:Default"];
    await context.Response.WriteAsync($"The config setting is: {defaultDebug}");
    string? environ = config["ASPNETCORE_ENVIRONMENT"];
    await context.Response.WriteAsync($" The env setting is: {environ}");
});
app.MapGet("/", async context => {
    await context.Response.WriteAsync("Hello World.");
});
app.Run();
```
x??

---

#### Changing the Environment in Launch Settings
The environment settings can be changed by modifying either `launchSettings.json` or specific launch configurations (e.g., in Visual Studio Code’s `launch.json`). This change will affect how configuration files are loaded and other runtime behaviors.
:p How do you change the `ASPNETCORE_ENVIRONMENT` setting in an ASP.NET Core application?
??x
You can change the `ASPNETCORE_ENVIRONMENT` setting by editing either the `launchSettings.json` file or the launch configurations in Visual Studio Code.

For `launchSettings.json`, you would modify it as follows:

```json
{
  "profiles": {
    "Platform": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": true,
      "applicationUrl": "http://localhost:5000",
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Production"
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

For Visual Studio Code, you would edit the `launch.json` file to change the environment variables:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": ".NET Core Launch (web)",
      "type": "coreclr",
      "request": "launch",
      "preLaunchTask": "build",
      "program": "${workspaceFolder}/bin/Debug/net7.0/Platform.dll",
      "args": [],
      "cwd": "${workspaceFolder}",
      "stopAtEntry": false,
      "serverReadyAction": {
        "action": "openExternally",
        "pattern": "\\bNow listening on:\\s+(https?://\\S+)"
      },
      "env": {
        "ASPNETCORE_ENVIRONMENT": "Production"
      },
      "sourceFileMap": {
        "/Views": "${workspaceFolder}/Views"
      }
    },
    {
      "name": ".NET Core Attach",
      "type": "coreclr",
      "request": "attach"
    }
  ]
}
```
x??

---

#### Using Environment Service in ASP.NET Core
Background context: The IWebHostEnvironment service is part of the ASP.NET Core platform and provides methods to determine the current environment. This avoids manually getting configuration settings, making the code cleaner and more maintainable.

:p How can you use the IWebHostEnvironment service to set up services during application initialization?
??x
You can access the `IWebHostEnvironment` through the `WebApplicationBuilder.Environment` property when setting up services. Here's an example of how to do this in `Program.cs`:

```csharp
var builder = WebApplication.CreateBuilder(args);
var servicesEnv = builder.Environment;
// - use environment to set up services
```

x??

---
#### Using Environment Service for Pipeline Configuration
Background context: The IWebHostEnvironment service can also be used when configuring the pipeline, via `WebApplication.Environment`.

:p How can you access the environment during the configuration of the application pipeline?
??x
You can access the `IWebHostEnvironment` through the `WebApplication.Environment` property when setting up the pipeline. Here's an example:

```csharp
var app = builder.Build();
var pipelineConfig = app.Configuration;
// - use configuration settings to set up pipeline
var pipelineEnv = app.Environment;
```

x??

---
#### Using Environment Service in Middleware Components
Background context: To access the environment within a middleware component or endpoint, you can define an `IWebHostEnvironment` parameter.

:p How do you pass the environment to a middleware component?
??x
You can define the `IWebHostEnvironment` as a parameter in your middleware class. Here's an example of how to use it in a middleware component:

```csharp
app.UseMiddleware<LocationMiddleware>();
app.MapGet("/config", async (HttpContext context, IConfiguration config, IWebHostEnvironment env) => {
    string? defaultDebug = config["Logging:LogLevel:Default"];
    await context.Response.WriteAsync($"The config setting is: {defaultDebug}");
    await context.Response.WriteAsync($" The env setting is: {env.EnvironmentName}");
});
```

x??

---
#### Storing User Secrets in ASP.NET Core
Background context: Sensitive data such as API keys, database connection passwords, or default administration accounts can be stored using the user secrets service. This ensures that sensitive information isn't accidentally committed to version control.

:p How do you initialize a project for storing user secrets?
??x
You initialize a project for storing user secrets by running the following command in your project directory:

```bash
dotnet user-secrets init
```

This adds an element to the `.csproj` file that contains a unique ID for the project, which is associated with the secrets on each developer's machine.

x??

---
#### Storing User Secrets Using CLI Commands
Background context: You can store specific user secrets using the `dotnet user-secrets set` command. Related secrets can be grouped by prefixing them with a common key.

:p How do you add a user secret to your project?
??x
You can add a user secret to your project using the following commands:

```bash
dotnet user-secrets set "WebService:Id" "MyAccount"
dotnet user-secrets set "WebService:Key" "MySecret123$"
```

Each command sets a key-value pair, and related secrets are grouped by using a common prefix.

x??

---
#### Listing User Secrets
Background context: After adding secrets, you can list them to verify that they have been stored correctly.

:p How do you list the user secrets for your project?
??x
You can list the user secrets for your project by running the following command:

```bash
dotnet user-secrets list
```

This will output a JSON-like representation of the secrets, showing each key-value pair.

x??

---
#### Using User Secrets in Configuration
Background context: User secrets are merged with normal configuration settings and accessed using `IConfiguration`.

:p How do you access user secrets within your application?
??x
You can access user secrets by injecting `IConfiguration` into your middleware or endpoints. Here's an example:

```csharp
app.MapGet("/config", async (HttpContext context, IConfiguration config) => {
    string? defaultDebug = config["Logging:LogLevel:Default"];
    await context.Response.WriteAsync($"The config setting is: {defaultDebug}");
    string? wsID = config["WebService:Id"];
    string? wsKey = config["WebService:Key"];
    await context.Response.WriteAsync($" The secret ID is: {wsID}");
    await context.Response.WriteAsync($" The secret Key is: {wsKey}");
});
```

x??

---
#### Setting the Development Environment
Background context: User secrets are loaded only when the application is set to the Development environment. You can change this setting in `launchSettings.json`.

:p How do you switch your application to the Development environment for using user secrets?
??x
You can change the environment to Development by editing the `launchSettings.json` file:

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
    "Platform": {
      "commandName": "Project",
      "dotnetRunMessages": true,
      "launchBrowser": true,
      "applicationUrl": "http://localhost:5000",
      "environmentVariables": {
        "ASPNETCORE_ENVIRONMENT": "Development"
      }
    },
    "IIS Express": {
      "commandName": "IISExpress",
      "launchBrowser": true,
      "environmentVariables": {
```

x??

---

#### Configuring User Secrets for ASP.NET Core
Background context: In this section, we learn how to use user secrets in an ASP.NET Core application. User secrets are used to store sensitive information like database connection strings or API keys that should not be checked into source control.

:p What is the purpose of using user secrets in an ASP.NET Core application?
??x
User secrets provide a way to securely store sensitive configuration data that is specific to the development environment and should not be committed to version control. This allows developers to keep their personal information, such as API keys or database connection strings, out of source code repositories.
x??

---

#### Logging in ASP.NET Core with User Secrets
Background context: This section covers how to configure logging for an ASP.NET Core application using user secrets and the console provider.

:p How can you generate logging messages in an ASP.NET Core application?
??x
In ASP.NET Core, you can generate logging messages by injecting the `ILogger<T>` service into your classes. The generic parameter T is used to specify the category of the log messages. For example:
```csharp
public class Population {
    public static async Task Endpoint(HttpContext context, ILogger<Population> logger) {
        // Log a debug message when processing starts
        logger.LogDebug("Started processing for {path}", context.Request.Path);
        
        // Process request logic here
        
        // Log another debug message when processing finishes
        logger.LogDebug("Finished processing for {path}", context.Request.Path);
    }
}
```
x??

---

#### Categories and Levels in Logging
Background context: This section explains the importance of categories and levels in logging within ASP.NET Core.

:p What are categories and levels in logging, and why are they important?
??x
Categories in logging help organize log messages by their source or type. In ASP.NET Core, you use the generic parameter T when injecting `ILogger<T>` to specify the category. Levels determine the severity of the message (e.g., Trace, Debug, Information, Warning, Error, Critical).

Levels are crucial for filtering and prioritizing log messages. For example:
```csharp
// Example of different logging levels
public class Population {
    public static async Task Endpoint(HttpContext context, ILogger<Population> logger) {
        logger.LogDebug("Started processing request");
        
        try {
            // Process request logic here
        } catch (Exception ex) {
            logger.LogError(ex, "Error occurred while processing request");
        }
        
        logger.LogInformation("Request processed successfully");
    }
}
```
x??

---

#### Logging in Program.cs File
Background context: This section explains how to log messages in the `Program.cs` file using the `Logger` property of the `WebApplication` class.

:p How do you log messages in the top-level statements of the `Program.cs` file?
??x
In the `Program.cs` file, you can use the `Logger` property provided by the `WebApplication` class to log messages. For example:
```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.Logger.LogDebug("Pipeline configuration starting");

// Application setup code here

app.Logger.LogDebug("Pipeline configuration complete");

app.Run();
```
x??

---

#### Console Provider for Logging
Background context: This section explains the use of the console provider as a simple logging mechanism.

:p Why is the console provider used in this chapter, and how does it work?
??x
The console provider is used because it is simple to set up and provides direct output to the terminal. It works by injecting `ILogger<T>` into your classes or methods and using its extension methods (e.g., `LogDebug`, `LogInformation`) to generate log messages that are then displayed in the console.

```csharp
public class Population {
    public static async Task Endpoint(HttpContext context, ILogger<Population> logger) {
        // Log a debug message when processing starts
        logger.LogDebug("Started processing for {path}", context.Request.Path);
        
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
        
        // Log a debug message when processing finishes
        logger.LogDebug("Finished processing for {path}", context.Request.Path);
    }
}
```
x??

---

