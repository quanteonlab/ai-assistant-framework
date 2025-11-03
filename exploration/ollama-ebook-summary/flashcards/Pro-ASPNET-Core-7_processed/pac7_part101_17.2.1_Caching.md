# Flashcards: Pro-ASPNET-Core-7_processed (Part 101)

**Starting Chapter:** 17.2.1 Caching data values

---

#### Caching Data Overview
Caching is a technique used to store frequently accessed data so that it can be retrieved faster, reducing the load on the system and improving performance. In this context, ASP.NET Core provides the `IDistributedCache` interface for caching data values across multiple instances of the application.

:p What is the purpose of using an IDistributedCache in ASP.NET Core?
??x
The purpose of using an IDistributedCache in ASP.NET Core is to store frequently accessed data in a way that allows it to be shared and accessed by multiple instances of the application. This helps in reducing the load on the system and improving performance.

```csharp
using Microsoft.Extensions.Caching.Distributed;
```
x??

---

#### Using Cache Service with SumEndpoint.cs
The `SumEndpoint` class is an example of how to use the cache service provided by ASP.NET Core to store calculated values. The endpoint calculates a sum from 1 to a given count and caches the result.

:p How does the `SumEndpoint` handle caching?
??x
The `SumEndpoint` checks if the cached value exists before calculating it. If no cached value is found, it performs the calculation and stores the result in the cache for future use. The cache key is generated based on the count parameter to ensure that different values are stored separately.

```csharp
int count;
int.TryParse((string?)context.Request.RouteValues["count"], out count);
string cacheKey = $"sum_{count}";
string? totalString = await cache.GetStringAsync(cacheKey);

if (totalString == null) {
    long total = 0;
    for (int i = 1; i <= count; i++) {
        total += i;
    }
    totalString = $"(DateTime.Now.ToLongTimeString()) {total}";
    await cache.SetStringAsync(cacheKey, totalString,
                               new DistributedCacheEntryOptions {
                                   AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(2)
                               });
}
await context.Response.WriteAsync(
    $"(DateTime.Now.ToLongTimeString()) Total for {count} values: {totalString}");
```
x??

---

#### Caching Service Setup in Program.cs
The `Program.cs` file sets up the cache service to use an in-memory cache. This is done by calling `AddDistributedMemoryCache`.

:p How does the `Program.cs` file configure the cache service?
??x
The `Program.cs` file configures the cache service to use an in-memory cache with a size limit of 200 items.

```csharp
var builder = WebApplication.CreateBuilder(args);
builder.Services.AddDistributedMemoryCache(opts => {
    opts.SizeLimit = 200;
});
```
x??

---

#### IDistributedCache Useful Methods
The `IDistributedCache` interface provides several useful methods for managing cached data. These methods allow you to store, retrieve, and manage the lifecycle of cached items.

:p What are some key methods of the `IDistributedCache` interface?
??x
Key methods of the `IDistributedCache` interface include:
- `GetString(key)`: Retrieves a string from the cache.
- `GetStringAsync(key)`: Asynchronously retrieves a string from the cache.
- `SetString(key, value, options)`: Stores a string in the cache with optional expiration settings.
- `SetStringAsync(key, value, options)`: Asynchronously stores a string in the cache with optional expiration settings.
- `Refresh(key)`: Resets the expiry interval for an item in the cache.
- `RefreshAsync(key)`: Asynchronously resets the expiry interval for an item in the cache.
- `Remove(key)`: Removes an item from the cache.
- `RemoveAsync(key)`: Asynchronously removes an item from the cache.

```csharp
string totalString = await cache.GetStringAsync(cacheKey);
if (totalString == null) {
    long total = 0;
    for (int i = 1; i <= count; i++) {
        total += i;
    }
    totalString = $"(DateTime.Now.ToLongTimeString()) {total}";
    await cache.SetStringAsync(cacheKey, totalString,
                               new DistributedCacheEntryOptions {
                                   AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(2)
                               });
}
```
x??

---

#### DistributedCacheEntryOptions Properties
The `DistributedCacheEntryOptions` class allows you to configure the behavior of cached items. Key properties include:
- `AbsoluteExpiration`: Specifies an absolute expiration date.
- `AbsoluteExpirationRelativeToNow`: Specifies a relative expiration time.
- `SlidingExpiration`: Specifies a period of inactivity after which the item will be ejected from the cache.

:p What are some important properties of the `DistributedCacheEntryOptions` class?
??x
Important properties of the `DistributedCacheEntryOptions` class include:
- `AbsoluteExpiration`: This property is used to specify an absolute expiry date. For example, you can set this to a future date and time.
- `AbsoluteExpirationRelativeToNow`: This property is used to specify a relative expiry period from the current moment.
- `SlidingExpiration`: This property is used to specify a period of inactivity after which the item will be ejected from the cache if it hasn't been read.

```csharp
await cache.SetStringAsync(cacheKey, totalString,
                           new DistributedCacheEntryOptions {
                               AbsoluteExpirationRelativeToNow = TimeSpan.FromMinutes(2)
                           });
```
x??

---

#### MemoryDistributedCacheOptions Configuration
The `MemoryDistributedCacheOptions` class is used to configure caching data in memory. This class provides several useful properties that control cache behavior, such as expiration scanning, size limits, and compaction.

:p What are the main properties of `MemoryDistributedCacheOptions`?

??x
- **ExpirationScanFrequency**: Sets a `TimeSpan` determining how often the cache scans for expired items.
- **SizeLimit**: Specifies the maximum number of items in the cache. When this limit is reached, the cache will start ejecting items.
- **CompactionPercentage**: Specifies the percentage by which the size of the cache is reduced when the `SizeLimit` is reached.

This configuration helps manage memory usage and ensures that the cache remains effective without consuming too many resources.

```csharp
builder.Services.AddDistributedMemoryCache(options =>
{
    options.SizeLimit = 200; // Limits the number of items to 200.
});
```
x??

---

#### Using Memory Cache for Caching Data
The `AddDistributedMemoryCache` method is used to add a distributed in-memory cache to an ASP.NET Core application. This cache stores data in memory as part of the ASP.NET Core process, which means it can be shared across multiple servers or containers but loses its contents when ASP.NET Core is stopped.

:p What are the limitations of using `AddDistributedMemoryCache` for caching?

??x
- The items stored in this cache are limited to the memory available within a single ASP.NET Core instance.
- If multiple instances of an application run on different servers or containers, they cannot share cached data because each stores its cache independently.
- When ASP.NET Core is stopped, the contents of the cache are lost.

This method is useful for caching data that does not need to be shared across multiple instances and can fit within memory constraints.

```csharp
app.MapGet("/sum", async context =>
{
    // Cache retrieval logic here
});
```
x??

---

#### Using DistributedSqlServerCache for Persistent Caching
The `AddDistributedSqlServerCache` method is used to add a distributed cache that stores data in a SQL Server database. This allows multiple instances of an ASP.NET Core application running on different servers or containers to share cached data, and the data persists even when ASP.NET Core is stopped.

:p How do you set up a persistent cache using `AddDistributedSqlServerCache`?

??x
1. **Create a Database**: Use SQL Server Management Studio or Azure Data Studio to create a new database named `CacheDb`.
2. **Install SQL Cache Package**: Use the following commands in PowerShell:
   ```csharp
   dotnet tool uninstall --global dotnet-sql-cache
   dotnet tool install --global dotnet-sql-cache --version 7.0.0
   ```
3. **Create Table**: Run the command to create a table in `CacheDb` named `DataCache`.
4. **Configure Cache Service**: Modify the service configuration in `Program.cs`:
   ```csharp
   builder.Services.AddDistributedSqlServerCache(options =>
   {
       options.ConnectionString = builder.Configuration["ConnectionStrings:CacheConnection"];
       options.SchemaName = "dbo";
       options.TableName = "DataCache";
   });
   ```

This setup ensures that cached data is stored persistently in a SQL Server database, allowing multiple instances of the application to share and access this cache.

```csharp
var app = builder.Build();
app.MapGet("/sum", async context =>
{
    // Cache retrieval logic here
});
```
x??

---

#### Creating a Persistent Cache Table
Creating a table for storing cached data in SQL Server involves running specific commands. These tables are used by the `AddDistributedSqlServerCache` method to store and retrieve cache items.

:p How do you create a persistent cache table using SQL?

??x
1. **Connect to Database**: Use the `sqlcmd` tool to connect to your LocalDB server:
   ```csharp
   sqlcmd -S "(localdb)\MSSQLLocalDB"
   ```
2. **Create Table Command**: Run the following commands in `sqlcmd`:
   ```sql
   CREATE DATABASE CacheDb;
   GO
   USE CacheDb;
   GO
   CREATE TABLE dbo.DataCache (Key NVARCHAR(450), Value NVARCHAR(MAX));
   GO
   ```

This creates a database named `CacheDb` and a table named `DataCache` within the `dbo` schema, which will be used to store cache items.

```sql
CREATE DATABASE CacheDb;
USE CacheDb;
CREATE TABLE dbo.DataCache (Key NVARCHAR(450), Value NVARCHAR(MAX));
```
x??

---

#### Configuring Caching Middleware
Background context: The provided text explains how to configure caching middleware using `IDistributedCache` and response caching. This involves setting up cache configurations and adding necessary services and middleware.

:p What is the purpose of configuring caching middleware?
??x
Configuring caching middleware helps in storing frequently accessed data or complete responses in a persistent manner, thus reducing the load on the application and improving performance by serving cached content instead of recomputing or retrieving from the database each time.
??x

---

#### Using IDistributedCache Service for Session Data
Background context: The text describes how to use `IDistributedCache` service to cache session-specific data. This involves setting up distributed caching with SQL Server and using session middleware.

:p How does `IDistributedCache` help in managing session data?
??x
`IDistributedCache` helps manage session data by storing it persistently, allowing the application to share session data across multiple instances or servers in a distributed environment.
??x

---

#### Configuring Response Caching
Background context: The text details how to configure and use response caching in ASP.NET Core. This involves adding `AddResponseCaching` service and middleware, setting up headers for cache control.

:p What is the role of the `AddResponseCaching` method?
??x
The `AddResponseCaching` method sets up the service used by the cache to manage cached responses. It needs to be called before any endpoint or middleware that requires response caching.
??x

---

#### Using Cache-Control Header for Response Caching
Background context: The text explains how to use the `Cache-Control` header in conjunction with response caching to control caching behavior.

:p How does the `Cache-Control` header enable response caching?
??x
The `Cache-Control` header enables response caching by specifying that responses can be cached if it includes the "public" directive and a maximum age. This header is crucial for telling the cache when to reuse or regenerate content.
??x

---

#### Generating HTML Responses with IResponseFormatter
Background context: The text demonstrates how to generate an HTML response using `IResponseFormatter` to ensure that browsers do not bypass the cache by including a zero max-age directive.

:p Why is generating an HTML response necessary for testing response caching?
??x
Generating an HTML response ensures that browsers include a Cache-Control header with a max-age of 0 when reloading, which bypasses the cache. Using `IResponseFormatter` to generate an HTML response and create a URL with `LinkGenerator` helps in reliably testing if the response cache is working.
??x

---

#### Implementing Response Compression
Background context: The text explains how to enable response compression for browsers that support it.

:p What is the purpose of adding `UseResponseCompression` middleware?
??x
Adding the `UseResponseCompression` middleware allows ASP.NET Core to compress responses sent to browsers that can handle compressed data, reducing bandwidth usage and potentially improving performance.
??x

---
Each flashcard is designed to cover a specific aspect of caching and response handling in ASP.NET Core. The questions are crafted to ensure familiarity with these concepts without requiring pure memorization.

