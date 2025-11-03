# Flashcards: Pro-ASPNET-Core-7_processed (Part 102)

**Starting Chapter:** 17.4 Caching output

---

#### Output Caching Overview
Output caching is a feature within ASP.NET Core designed to improve performance by storing and serving cached copies of responses for certain HTTP requests. This approach is more flexible than response caching, but it also introduces complexity due to the need for careful locking when handling cached data.

Caching is applied at the endpoint level rather than on individual HTTP responses, making it suitable for scenarios where the same request could be served from a cache multiple times without recalculating results.

:p What is output caching in ASP.NET Core?
??x
Output caching in ASP.NET Core stores and serves cached copies of responses for specific endpoints. This improves performance by reducing the need to recalculate results for each request, but it requires careful handling due to potential concurrency issues with cached data.
x??

---

#### Updating MapEndpoint<T> Extension Method
The `MapEndpoint<T>` extension method was updated to produce an implementation that can be used with output caching. It uses reflection to invoke methods and creates instances of endpoints.

:p How does the updated `MapEndpoint<T>` method work?
??x
The updated `MapEndpoint<T>` method now returns an `IEndpointConventionBuilder` instance, which allows for endpoint configuration including output caching. Here's a simplified version:

```csharp
using System.Reflection;
namespace Microsoft.AspNetCore.Builder {
    public static class EndpointExtensions {
        public static IEndpointConventionBuilder MapEndpoint<T>(
            this IEndpointRouteBuilder app,
            string path,
            string methodName = "Endpoint"
        ) {
            MethodInfo? methodInfo = typeof(T).GetMethod(methodName);
            if (methodInfo?.ReturnType != typeof(Task)) {
                throw new System.Exception("Method cannot be used");
            }
            ParameterInfo[] methodParams = methodInfo.GetParameters();
            return app.MapGet(path, context => {
                T endpointInstance = 
                    ActivatorUtilities.CreateInstance<T>(context.RequestServices);
                return (Task)methodInfo.Invoke(
                    endpointInstance,
                    methodParams.Select(p => 
                        p.ParameterType == typeof(HttpContext) 
                        ? context 
                        : context.RequestServices.GetService(p.ParameterType)
                    ).ToArray()
                );
            });
        }
    }
}
```

This extension method uses reflection to invoke the specified method on an instance of `T`, passing appropriate parameters from the request or service provider.

x??

---

#### Implementing Output Caching in Program.cs
The configuration for output caching is set up by adding services and middleware, then applying it to specific endpoints. This ensures that cached results are used when possible, reducing the load on the application server.

:p How does the output caching configuration work in `Program.cs`?
??x
Output caching is configured by calling `AddOutputCache`, registering the caching middleware with `UseOutputCache`, and enabling caching for a specific endpoint using `CacheOutput`.

```csharp
builder.Services.AddOutputCache();
app.UseOutputCache();

// Enable caching for the sum endpoint.
app.MapEndpoint<Platform.SumEndpoint>("/sum/{count:int=1000000000}")
    .CacheOutput();
```

This configuration ensures that requests to the `/sum` endpoint will be cached, reducing the computational load on each request.

x??

---

#### Disabling Cache-Control Header
To prevent the `Cache-Control` header from being set directly in the response, the relevant code was commented out. This allows the output caching mechanism to handle caching without interference.

:p Why is the cache-control header disabled in the endpoint?
??x
Disabling the `Cache-Control` header prevents it from being explicitly set by the endpoint logic. Instead, the built-in output caching mechanism handles caching based on its configured policies. This separation of concerns ensures that changes to caching can be made without modifying the response-generating code.

```csharp
//context.Response.Headers["Cache-Control"] = "public, max-age=120";
```

This line is commented out, meaning it does not set any `Cache-Control` headers directly.

x??

---

#### Default Caching Policy
The default caching policy used in the example caches content for one minute and applies to HTTP GET or HEAD requests that produce a 200 response. It excludes authenticated requests and those that set cookies.

:p What is the default caching policy applied in the provided configuration?
??x
The default caching policy in the provided configuration caches responses for 1 minute (60 seconds) under the following conditions:
- The request method must be either GET or HEAD.
- The response status code should be 200 OK.
- Requests must not be authenticated.
- Responses must not set any cookies.

This ensures that the cache is only used for non-authenticated, read-only requests to reduce server load and improve performance.

x??

---

---
#### Caching Output Policy Definition
Caching policies can be customized to apply different caching strategies based on specific endpoints. This is done using an options pattern, which allows for flexible configuration of cache behavior.

:p How are custom caching policies defined and applied to specific endpoints?
??x
Custom caching policies are defined using the `AddOutputCache` method from the `Platform.Services` namespace in the `Program.cs` file. You can define a base policy that applies globally or create specific policies for different endpoints.

For example, consider the following code snippet:

```csharp
builder.Services.AddOutputCache(opts => {
    opts.AddBasePolicy(policy => {
        policy.Cache();
        policy.Expire(TimeSpan.FromSeconds(10));
    });
    opts.AddPolicy("30sec", policy => {
        policy.Cache();
        policy.Expire(TimeSpan.FromSeconds(30));
    });
});
```

Here, a base policy sets the cache duration to 10 seconds for all requests by default. A custom `30sec` policy is also defined, which expires cached content after 30 seconds.

To apply these policies to specific endpoints, use the `CacheOutput` method:

```csharp
app.MapEndpoint<Platform.SumEndpoint>("/sum/{count:int=1000000000}")
    .CacheOutput();

app.MapEndpoint<Platform.SumEndpoint>("/sum30/{count:int=1000000000}")
    .CacheOutput("30sec");
```

This ensures that requests to `/sum` are cached with the default 10-second policy, while requests to `/sum30` use the custom `30sec` policy.

x??
---

#### Installing Entity Framework Core Global Tool Package
Background context: This section describes how to install and test the Entity Framework Core (EF Core) global tool package, which is necessary for managing databases from the command line and handling data access packages within a project. The package includes commands like `dotnet ef` that are essential for working with EF Core.

:p How do you install the Entity Framework Core global tool package?
??x
To install the Entity Framework Core global tool package, first remove any existing versions using:
```shell
dotnet tool uninstall --global dotnet-ef
```
Then, install a specific version (e.g., 7.0.0) with:
```shell
dotnet tool install --global dotnet-ef --version 7.0.0
```
This command installs the necessary tools to manage EF Core databases and projects.

To test if the package is installed correctly, run:
```shell
dotnet ef --help
```
The output should provide information on available commands like `database`, `dbcontext`, and `migrations`.

x?

---

#### Adding Entity Framework Core Packages to the Project
Background context: This section explains how to add necessary EF Core packages to a project. These packages are required for creating, managing databases, and working with data models.

:p How do you add necessary Entity Framework Core packages to your project?
??x
To add the required packages using `dotnet` commands in Visual Studio Code or from the command line, navigate to the project folder containing the `Platform.csproj` file and run:
```shell
dotnet add package Microsoft.EntityFrameworkCore.Design --version 7.0.0
dotnet add package Microsoft.EntityFrameworkCore.SqlServer --version 7.0.0
```
These commands install packages needed for designing your database context and interacting with SQL Server databases.

x?

---

#### Creating the Data Model in C#
Background context: This section details how to create a simple data model using C# classes, which will be used by Entity Framework Core to manage the corresponding database schema. The `Calculation` class defines properties representing a calculation's ID, count, and result.

:p How do you define a simple data model for calculations in C#?
??x
You define a simple data model by creating a class named `Calculation` with relevant properties:
```csharp
namespace Platform.Models {
    public class Calculation {
        public long Id { get; set; }
        public int Count { get; set; }
        public long Result { get; set; }
    }
}
```
This class will be used by Entity Framework Core to map the data model to a database schema.

x?

---

#### Creating the DbContext Class
Background context: The `CalculationContext` class is derived from `DbContext`, which provides access to the database. This class defines how calculations are stored and retrieved in the database through the `Calculations` property, which represents a set of `Calculation` objects.

:p How do you create a DbContext class for managing calculations?
??x
To manage calculations using Entity Framework Core, create a `DbContext` subclass named `CalculationContext`:
```csharp
using Microsoft.EntityFrameworkCore;

namespace Platform.Models {
    public class CalculationContext : DbContext {
        public CalculationContext(
            DbContextOptions<CalculationContext> opts) : base(opts) { }
        public DbSet<Calculation> Calculations => Set<Calculation>();
    }
}
```
This class defines a constructor to initialize the context and a `DbSet` property for managing collections of `Calculation` objects.

x?

---

#### Configuring Database Service in Program.cs
Background context: This section explains how to configure the database service within the `Program.cs` file. It includes setting up connection strings, configuring caching policies, and adding data access services using EF Core commands.

:p How do you set up a DbContext for accessing calculations in your application?
??x
In the `Program.cs` file, add the following configuration:
```csharp
builder.Services.AddDbContext<CalculationContext>(opts => {
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:CalcConnection"]);
});
```
This line configures EF Core to use SQL Server and connects to the database specified by the "CalcConnection" string in `appsettings.json`.

x?

---

