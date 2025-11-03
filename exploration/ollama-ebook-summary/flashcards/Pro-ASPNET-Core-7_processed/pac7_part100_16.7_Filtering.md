# Flashcards: Pro-ASPNET-Core-7_processed (Part 100)

**Starting Chapter:** 16.7 Filtering requests using the host header

---

---
#### Using Status Code Middleware in ASP.NET Core
Background context explaining how status codes are handled using middleware. The UseStatusCodePages method is used to intercept and handle specific HTTP status codes, adding custom responses based on these codes. This is useful for logging, providing user-friendly messages, or redirecting.
:p How does the `UseStatusCodePages` method work in ASP.NET Core?
??x
The `UseStatusCodePages` method configures middleware that handles HTTP status codes between 400 and 600. It intercepts these responses and adds a custom response body based on the provided template string, which can include placeholders for the status code.
```csharp
app.UseStatusCodePages("text/html", "<h1>Status: %Error</h1>");
```
x??

---
#### Handling Unhandled Exceptions with Status Code Middleware
Background context explaining that exceptions disrupt request processing and preventing middleware from inspecting responses before they are sent. As a result, `UseStatusCodePages` is typically used in conjunction with other methods like `UseExceptionHandler` or `UseDeveloperExceptionPage`.
:p How does the UseStatusCodePages method handle unhandled exceptions?
??x
The `UseStatusCodePages` method will not respond to unhandled exceptions because these disrupt the flow of the request through the pipeline, preventing the middleware from inspecting and altering the response before it is sent to the client. Instead, you should use `UseExceptionHandler` or `UseDeveloperExceptionPage` to handle such cases.
```csharp
app.UseExceptionHandler("/error.html");
```
x??

---
#### Using StatusCode Middleware with Redirects and Re-Execution
Background context explaining that there are related methods like `UseStatusCodePagesWithRedirects` and `UseStatusCodePagesWithReExecute`, which redirect the client or re-run the request with a different URL. These methods can lead to the original status code being lost.
:p How do `UseStatusCodePagesWithRedirects` and `UseStatusCodePagesWithReExecute` work?
??x
These methods handle responses by either redirecting the client to a different URL (`UseStatusCodePagesWithRedirects`) or re-running the request through the pipeline with a different URL (`UseStatusCodePagesWithReExecute`). Both can cause the original status code to be lost, as they may send new responses that overwrite the initial one.
```csharp
app.UseStatusCodePagesWithRedirects("/new-url/{0}");
```
x??

---
#### Filtering Requests Using Host Header Middleware
Background context explaining how the HTTP specification requires a Host header for specifying the intended hostname. The default middleware in ASP.NET Core filters requests based on this header, allowing only those targeting approved hostnames.
:p What does the `AllowedHosts` configuration property do?
??x
The `AllowedHosts` configuration property in `HostFilteringOptions` specifies which hostnames are allowed to send requests to your application. It returns a List<string> of domains that can be matched using wildcards, such as `*.example.com`. If this list is empty or includes `"*"` (the default value), any hostname will be accepted.
```csharp
builder.Services.Configure<HostFilteringOptions>(opts => {
    opts.AllowedHosts.Clear();
    opts.AllowedHosts.Add("*.example.com");
});
```
x??

---

#### Caching Data Values
Background context: This section explains how to cache data values in ASP.NET Core, which is useful for improving efficiency by avoiding re-creation of expensive-to-produce data. It involves setting up a caching service that can store and retrieve cached data across requests.

:p How does caching data values work in ASP.NET Core?
??x
Caching data values works by storing frequently accessed or expensive-to-produce data in memory, on disk, or using an external cache system. This reduces the need to regenerate the same data for each request, improving performance and resource usage. Caching services can be set up at both endpoint and middleware levels.

For example:
- An endpoint might store the results of a complex database query.
- Middleware could intercept requests and serve cached responses if they exist.

```csharp
public class CacheService {
    private readonly IMemoryCache _cache;

    public CacheService(IMemoryCache cache) {
        _cache = cache;
    }

    public T GetOrAdd<T>(string key, Func<T> factory) where T : notnull {
        return _cache.GetOrCreate(key, entry => factory());
    }
}
```

x??

---

#### Creating a Persistent Cache
Background context: A persistent cache uses the database to store cached data, providing longer-term storage compared to in-memory caching. This is useful for scenarios where data needs to be retained even after an application restart.

:p How can you set up a persistent cache using SQL Server LocalDB?
??x
To set up a persistent cache with SQL Server LocalDB, you need to configure the `IMemoryCache` service to use a database-backed storage mechanism. This typically involves configuring Entity Framework Core to interact with the database and storing cache entries in tables.

For example:
- Define an entity for cache entries.
- Use Entity Framework Core to manage the lifecycle of these entities.

```csharp
public class CacheEntry {
    public int Id { get; set; }
    public string Key { get; set; }
    public string Value { get; set; }
}

public class ApplicationDbContext : DbContext {
    public DbSet<CacheEntry> CacheEntries { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder) {
        optionsBuilder.UseSqlServer("Your connection string");
    }
}
```

x??

---

#### Caching Entire Responses
Background context: ASP.NET Core provides middleware for caching entire responses, which can be controlled using the `Cache-Control` header. This is useful for reducing load on servers by serving cached content when possible.

:p How does response or output caching work in ASP.NET Core?
??x
Response or output caching works by storing the complete HTTP response generated by an endpoint and returning it directly from cache if conditions are met. The middleware intercepts requests, checks the `Cache-Control` header for instructions, and serves cached responses if they match the criteria.

For example:
- Configure middleware to enable response caching.
- Set appropriate cache settings such as duration and vary-by headers.

```csharp
public void ConfigureServices(IServiceCollection services) {
    services.AddResponseCaching();
}

public void Configure(IApplicationBuilder app) {
    app.UseResponseCaching();

    // Example configuration for a specific endpoint
    app.Use(async (context, next) => {
        if (context.Request.Path == "/cached-endpoint") {
            context.Response.GetTypedHeaders().CacheControl =
                new CacheControlHeaderValue {
                    Public = true,
                    MaxAge = TimeSpan.FromDays(7)
                };
            await next();
        }
    });
}
```

x??

---

#### Storing Application Data
Background context: This section covers accessing and storing application data using Entity Framework Core. It involves setting up a database schema, creating migrations, and consuming the `DbContext` service in endpoints.

:p How do you set up a database for storing application data in ASP.NET Core?
??x
To set up a database for storing application data in ASP.NET Core, follow these steps:

1. Define entities that represent your data model.
2. Create a DbContext to interact with the database.
3. Configure Entity Framework Core to connect to the database using connection strings.
4. Apply migrations to create and update the schema.

For example:
- Define an entity for a simple item.
- Set up migration scripts.
- Apply migrations to the database.

```csharp
public class Item {
    public int Id { get; set; }
    public string Name { get; set; }
}

public class ApplicationDbContext : DbContext {
    public DbSet<Item> Items { get; set; }

    protected override void OnConfiguring(DbContextOptionsBuilder optionsBuilder) {
        optionsBuilder.UseSqlServer("Your connection string");
    }
}
```

x??

---

#### Creating a Database Schema
Background context: Creating and applying migrations is crucial for managing the database schema in ASP.NET Core projects. This ensures that your application's data model remains consistent with the database structure.

:p How do you create a database schema using Entity Framework Core?
??x
To create a database schema using Entity Framework Core, follow these steps:

1. Define your entity models.
2. Create an `DbContext` class to manage interactions with the database.
3. Use migration tools to generate and apply scripts that create or update the schema.

For example:
- Generate initial migrations.
- Apply migrations to the current state of the database.

```csharp
// Initial migration setup
Add-Migration InitialCreate

// Apply migration
Update-Database
```

x??

---

#### Accessing Data in Endpoints
Background context: To access data stored in a database, you need to consume the `DbContext` service provided by Entity Framework Core. This involves setting up dependency injection and using LINQ queries to interact with the database.

:p How do you consume a database context service in an ASP.NET Core endpoint?
??x
To consume a database context service in an ASP.NET Core endpoint, follow these steps:

1. Register `DbContext` as a scoped service in `ConfigureServices`.
2. Inject `DbContext` into your controller or endpoint.
3. Use LINQ queries to interact with the database.

For example:
- Register services in `Startup.cs`.

```csharp
public void ConfigureServices(IServiceCollection services) {
    services.AddDbContext<ApplicationDbContext>(options =>
        options.UseSqlServer(Configuration.GetConnectionString("DefaultConnection")));
}
```

- Consume the service in a controller or endpoint.

```csharp
[ApiController]
[Route("[controller]")]
public class ItemsController : ControllerBase {
    private readonly ApplicationDbContext _context;

    public ItemsController(ApplicationDbContext context) {
        _context = context;
    }

    [HttpGet]
    public async Task<ActionResult<IEnumerable<Item>>> GetItems() {
        return await _context.Items.ToListAsync();
    }
}
```

x??

---

#### Including All Request Details in Logging Messages
Background context: Ensuring that all request details are included in logging messages can help with debugging and monitoring. The sensitive data logging feature ensures that sensitive information is logged appropriately.

:p How do you enable sensitive data logging in ASP.NET Core?
??x
To enable sensitive data logging in ASP.NET Core, use the `SensitiveDataLogging` middleware to capture and log request details such as headers and query strings without exposing sensitive information.

For example:
- Configure the middleware in `Startup.cs`.

```csharp
public void ConfigureServices(IServiceCollection services) {
    // Other service configurations...
}

public void Configure(IApplicationBuilder app, IWebHostEnvironment env) {
    if (env.IsDevelopment()) {
        app.UseDeveloperExceptionPage();
        app.UseSensitiveDataLogging();
    }

    // Other middleware registrations...
}
```

x??

#### Replacing Program.cs Contents
Background context: In this chapter, you are working with a project called Platform from Chapter 16. The goal is to replace the contents of the `Program.cs` file to set up a basic ASP.NET Core application that returns "Hello World" when accessed at the root URL.

:p What does replacing the content of the Program.cs file accomplish?
??x
Replacing the content of the `Program.cs` file sets up a simple HTTP endpoint that responds with "Hello World". This is the first step in demonstrating more complex operations, such as caching and data handling. The updated code looks like this:

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapGet("/", async context =>
{
    await context.Response.WriteAsync("Hello World.");
});

app.Run();
```

x??

---

#### Starting the ASP.NET Core Runtime
Background context: To test the setup, you need to start the ASP.NET Core application using a command prompt. This involves navigating to the project folder and running `dotnet run`.

:p How do you start the ASP.NET Core runtime?
??x
To start the ASP.NET Core runtime, open a new PowerShell command prompt, navigate to the Platform project folder containing the `Platform.csproj` file, and run the following command:

```cmd
dotnet run
```

This command compiles the application and runs it on your local machine.

x??

---

#### Adding an Endpoint for Data Caching
Background context: To simulate data caching, you add a class that performs expensive computations. The `SumEndpoint` class calculates the sum of a series of integers, which simulates expensive computation. This is done to understand how caching can improve performance in web applications.

:p What does the SumEndpoint.cs file do?
??x
The `SumEndpoint.cs` file contains a class named `SumEndpoint`. It defines an asynchronous method that calculates the sum of a sequence of numbers and writes the result to the HTTP response. The code looks like this:

```csharp
namespace Platform {
    public class SumEndpoint {
        public async Task Endpoint(HttpContext context) {
            int count;
            int.TryParse((string?)context.Request.RouteValues["count"], out count);
            long total = 0;
            for (int i = 1; i <= count; i++) {
                total += i;
            }
            string totalString = $"({DateTime.Now.ToLongTimeString()}) " + total;
            await context.Response.WriteAsync($"({DateTime.Now.ToLongTimeString()}) Total for {count} values: {totalString}");
        }
    }
}
```

This class simulates an expensive computation by summing a large number of integers and then returns the result.

x??

---

#### Mapping the Endpoint in Program.cs
Background context: To make the `SumEndpoint` accessible via HTTP, you map it to a specific route using the `MapEndpoint` method. This allows clients to request this endpoint via a URL.

:p How do you map the SumEndpoint class to an HTTP route?
??x
You can map the `SumEndpoint` class to an HTTP route by adding the following line in the `Program.cs` file:

```csharp
app.MapEndpoint<Platform.SumEndpoint>("/sum/{count:int=1000000000}");
```

This line maps the `SumEndpoint` to a URL path where `{count}` is a parameter that defaults to 1,000,000,000 if not provided. The full file looks like this:

```csharp
var builder = WebApplication.CreateBuilder(args);
var app = builder.Build();

app.MapEndpoint<Platform.SumEndpoint>("/sum/{count:int=1000000000}");
app.MapGet("/", async context =>
{
    await context.Response.WriteAsync("Hello World.");
});

app.Run();
```

x??

---

#### Understanding Endpoint Behavior
Background context: The `SumEndpoint` performs a computation-intensive task and writes the result to the HTTP response. When you request this endpoint, it recalculates the sum each time.

:p What behavior does the SumEndpoint have when accessed multiple times?
??x
The `SumEndpoint` recalculates the sum of integers every time it is accessed. This means that if you reload the browser window after accessing `/sum`, the calculation will be performed again, as indicated by different timestamps in the response. For example:

```
(07:34:21) Total for 1000000000 values: (1589693747) 500000000500000000
(07:34:24) Total for 1000000000 values: (1589693747) 500000000500000000
```

This shows that the result is produced fresh for each request, even though the URL and parameters are identical.

x??

---

