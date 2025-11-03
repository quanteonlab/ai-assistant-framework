# Flashcards: Pro-ASPNET-Core-7_processed (Part 108)

**Starting Chapter:** 20.1.1 Dropping the database

---

---
#### RESTful Web Services Overview
Background context: This section provides an overview of how RESTful web services are structured and operate using HTTP methods and URLs. It mentions that controllers are typically used to create these services, which can significantly improve project scalability.

:p What are the key components of a RESTful web service as described?
??x
RESTful web services use the HTTP method (e.g., GET, POST) and URL to specify operations performed on resources. Controllers are often employed for creating web services due to their ability to manage complex business logic efficiently.
x??

---
#### Controller Basics in ASP.NET Core
Background context: The controller is a fundamental part of an ASP.NET Core application that handles incoming requests by performing specific tasks or actions.

:p What role do controllers play in ASP.NET Core applications?
??x
Controllers are responsible for receiving HTTP requests, processing them according to the business logic defined within the action methods, and then returning appropriate responses. They act as a bridge between the client's request and the application's data model.
x??

---
#### Action Methods with Attributes
Background context: Action methods in ASP.NET Core controllers can be decorated with attributes to specify which HTTP methods they should respond to.

:p How do you decorate an action method to accept specific HTTP methods?
??x
You use attributes such as `[HttpGet]`, `[HttpPost]`, etc., to indicate the HTTP method that the action method will handle. For example:
```csharp
[HttpGet]
public IActionResult GetItems()
{
    // Logic for handling GET request
}
```
x??

---
#### Model Binding in ASP.NET Core
Background context: Model binding is a process where ASP.NET Core automatically maps incoming request data to objects passed as parameters to action methods.

:p How does model binding work in ASP.NET Core?
??x
Model binding extracts data from the HTTP request and passes it to an action method as strongly-typed objects. This simplifies handling form inputs, JSON payloads, etc., by directly using them within your application logic.
```csharp
[HttpPost]
public IActionResult CreateItem([FromBody] Item item)
{
    // Logic for creating a new item
}
```
x??

---
#### Data Validation in Model Binding
Background context: Validating data after model binding ensures that incoming data meets certain criteria before being processed by the application.

:p How can you perform data validation on bound models?
??x
You can use attributes like `[Required]`, `[Range]`, etc., to validate properties of your model objects. ASP.NET Core automatically applies these validations, and if any are violated, it returns a 400 Bad Request response with validation errors.
```csharp
public class Item {
    [Required]
    public string Name { get; set; }
}
```
x??

---
#### Rate Limiting in Web Services
Background context: Chapter 16 covered rate limiting, which can be applied to web services to prevent abuse and ensure fair usage.

:p How can you apply rate limiting to a web service?
??x
You implement rate limiting by configuring middleware or using specific attributes that restrict the number of requests allowed within a certain time frame. For instance:
```csharp
services.AddCaching(options => {
    options.EnableRateLimiting = true;
});
```
This configuration enables rate limiting for caching.
x??

---
#### Advanced Web Service Features Overview
Background context: This section introduces advanced features such as managing related data, supporting the PATCH method, and content negotiation.

:p What are some advanced web service features discussed in this chapter?
??x
Advanced features include:
- Managing related data using Entity Framework Core queries.
- Supporting the HTTP PATCH method for selective updates.
- Content negotiation to support different response formats.
- Caching output from web services.
- Generating documentation for a web service.
x??

---
#### Managing Related Data with EF Core
Background context: This feature allows you to include and manage related data in web service responses using LINQ queries.

:p How do you manage related data in Entity Framework Core?
??x
You use `Include` and `ThenInclude` methods to load related entities. For example:
```csharp
public IActionResult GetSuppliers()
{
    var suppliers = context.Suppliers.Include(s => s.Items)
                                      .ThenInclude(i => i.Category);
    return Ok(suppliers);
}
```
This ensures that related data is loaded along with the primary entity.
x??

---
#### Supporting HTTP PATCH Method
Background context: The PATCH method allows for partial updates to resources, making it useful for selective modifications.

:p How do you support the HTTP PATCH method in an ASP.NET Core application?
??x
You decorate your action methods with `[HttpPatch]` and handle partial updates appropriately. For example:
```csharp
[HttpPatch]
public IActionResult UpdateItem([FromBody] Item item)
{
    // Logic to update parts of the item
}
```
This allows you to update specific fields of an entity without sending all data.
x??

---
#### Content Negotiation in Web Services
Background context: Content negotiation enables web services to return responses in various formats based on client preferences.

:p How can content negotiation be used in ASP.NET Core?
??x
You use `Accept` headers or `[Produces]` attributes to specify acceptable response types. For example:
```csharp
[HttpGet]
[Produces("application/json", "text/html")]
public IActionResult GetItems()
{
    // Logic to return items
}
```
This allows clients to request responses in JSON, HTML, etc.
x??

---
#### Caching Web Service Responses
Background context: Caching can improve performance by storing and reusing previously computed results.

:p How do you cache web service responses in ASP.NET Core?
??x
You use the `OutputCache` attribute or configure caching middleware. For example:
```csharp
[HttpGet]
[OutputCache(Duration = 300, VaryByQueryString = "id")]
public IActionResult GetItem(int id)
{
    // Logic to fetch item
}
```
This caches the response for 5 minutes based on the `id` parameter.
x??

---
#### Documenting Web Services with OpenAPI
Background context: Describing web services through documentation helps in understanding their functionality and usage.

:p How do you document a web service using OpenAPI?
??x
You use OpenAPI (formerly known as Swagger) to generate API descriptions. This involves creating a `Startup.cs` configuration or using tools like Swashbuckle.
```csharp
services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new Info { Title = "My Web Service", Version = "v1" });
});
```
This setup allows you to describe the service's endpoints, request/response formats, etc., making it easier for developers to understand and consume.
x??

---

#### Downloading Example Project
Background context: The chapter provides a way to download example projects from GitHub, making it easier for readers to follow along with practical examples. This ensures that users can test and learn from real-world scenarios.

:p How do you access the example project for this chapter?
??x
To access the example project, go to the URL provided in the book: https://github.com/manningbooks/pro-asp.net-core-7. From there, you can download or clone the repository depending on your preference.
x??

---

#### Dropping the Database
Background context: The text explains how to drop the database using Entity Framework Core commands. This is a common operation when preparing an environment for testing and development.

:p How do you drop the database in ASP.NET Core?
??x
To drop the database, open a new PowerShell command prompt, navigate to the folder containing the `WebApp.csproj` file, and run:
```shell
dotnet ef database drop --force
```
This command will remove the current database schema.
x??

---

#### Running the Example Application
Background context: The chapter details how to run the example application after making necessary changes like dropping the database. This step is crucial for ensuring that the application functions correctly.

:p How do you start running the example application?
??x
To start the example application, open a new PowerShell command prompt and navigate to the folder containing the `WebApp.csproj` file. Then run:
```shell
dotnet run
```
This command compiles and runs the application, which seeds the database as part of its startup process.
x??

---

#### Dealing with Related Data (Part 1)
Background context: The text explains that Entity Framework Core includes navigation properties in data model classes, allowing for efficient querying of related data. However, this can lead to issues such as circular references.

:p What is the issue when using Include() method in Entity Framework Core?
??x
When you use the `Include()` method to eager load related entities, it can introduce a circular reference because Entity Framework Core will populate both the parent and child navigation properties. This leads to an object cycle during serialization, causing errors.
x??

---

#### Dealing with Related Data (Part 2)
Background context: The text highlights that the Include() method is useful in some applications but problematic in ASP.NET Core due to how database contexts are managed.

:p Why does using the Include() method cause issues in ASP.NET Core?
??x
In ASP.NET Core, a new `DbContext` is created for each HTTP request. When you use `Include()` with Entity Framework Core, it loads related entities and also populates their navigation properties back to the parent entity. This results in circular references, as both sides of the relationship are being serialized, leading to an "object cycle" error.
x??

---

#### Handling Circular References
Background context: The text discusses how to handle circular references when serializing data with JSON. It explains that ignoring null values is not always sufficient and suggests alternative methods.

:p How can you handle circular references in Entity Framework Core?
??x
To handle circular references, you need to modify the way related entities are loaded or serialized. One approach is to use DTOs (Data Transfer Objects) instead of directly exposing your entity models. Another method is to configure the JSON serializer to ignore cycles.

Example with DTO:
```csharp
public class SupplierDto {
    public long Id { get; set; }
    public string Name { get; set; }
    // other properties without navigation properties
}

[HttpGet("{id}")]
public async Task<Supplier?> GetSupplier(long id) {
    return await context.Suppliers
        .Include(s => s.Products)
        .Select(s => new SupplierDto {
            Id = s.Id,
            Name = s.Name,
            // map other properties as needed, avoiding circular references
        })
        .FirstAsync(s => s.Id == id);
}
```
x??

---

#### Circular References in Entity Framework Core
Background context explaining the issue of circular references when using Entity Framework Core. This occurs because EF Core fetches related entities and follows the navigation properties, which can lead to loops if not managed correctly.

Entity Framework Core loads related data by following navigation properties between entities. When a `Supplier` object is returned by the controller’s action method, it includes its associated `Product` objects. Each `Product` object also references back to the `Supplier`, creating a circular reference loop until reaching the maximum depth limit set by EF Core.

:p How does Entity Framework Core handle related data and what issue can arise from this?
??x
Entity Framework Core follows navigation properties between entities when loading related data, which can result in circular references if not properly managed. This happens because each `Supplier` object includes its associated `Product` objects, and each `Product` has a reference back to the `Supplier`, creating an infinite loop.

To manage this, you need to break these circular references before serializing the JSON data. This is typically done by modifying the objects after they have been created by EF Core and before they are serialized.
??x
The answer with detailed explanations.
To address this issue, you can modify the object graph to remove or handle circular references. For instance, in your `SuppliersController`, you might adjust the `Supplier` entity's properties or use a projection (selecting only necessary fields) to avoid including certain navigation properties that cause loops.

Here is an example of how you could project and return a simplified version of the `Supplier` object:
```csharp
[HttpGet("{id}")]
public async Task<Supplier?> GetSupplier(long id)
{
    var supplier = await context.Suppliers
        .Where(s => s.Id == id) // Fetch the supplier by ID
        .Select(s => new Supplier 
        { 
            Id = s.Id, 
            Name = s.Name,
            Products = null // Exclude products to avoid circular reference
        })
        .FirstOrDefaultAsync();
    return supplier;
}
```
By excluding `Products` from the projection, you prevent EF Core from loading this navigation property and thus break any potential circular references.

You can also manually remove or adjust properties before serialization if needed:
```csharp
public class Supplier 
{
    public long Id { get; set; }
    public string Name { get; set; } = "";
    public List<Product> Products { get; set; } = new();
}

// In your controller method, adjust the object graph before returning it.
```
This ensures that when the JSON serializer processes the data, it does not encounter circular references.

```csharp
public class SuppliersController : ControllerBase 
{
    private DataContext context;

    public SuppliersController(DataContext ctx) 
    {
        context = ctx;
    }

    [HttpGet("{id}")]
    public async Task<Supplier?> GetSupplier(long id)
    {
        var supplier = await context.Suppliers
            .Where(s => s.Id == id)
            .Select(s => new Supplier 
            { 
                Id = s.Id, 
                Name = s.Name,
                Products = null // Excluding products to avoid circular reference
            })
            .FirstOrDefaultAsync();
        return supplier;
    }
}
```
x??

