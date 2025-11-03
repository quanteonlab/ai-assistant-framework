# Flashcards: Pro-ASPNET-Core-7_processed (Part 107)

**Starting Chapter:** 19.5.5 Applying the API controller attribute

---

#### Concept of Validation and Error Handling in ASP.NET Core Web Services
Background context: This concept explains how to handle client data validation in an ASP.NET Core web service. When a request is received, if the data is valid, an `Ok` response is returned. If not, a `BadRequest` response with details of the validation errors is sent back.

:p How does validation work in an ASP.NET Core web service when handling POST requests?
??x
In ASP.NET Core, validation is performed on client-sent data before processing it further. If the data is valid according to the specified rules (e.g., ranges for numeric fields), a successful `Ok` response is returned. However, if the data fails validation checks, the `IsValid` property will be false and a `BadRequest` response is sent with details of the errors.

Code Example:
```csharp
if (!ModelState.IsValid)
{
    return BadRequest(ModelState);
}
```
x??

---

#### Concept of Model Binding in ASP.NET Core Web Services
Background context: This concept covers how data from the request body is bound to a model and validated. The `FromBody` attribute is used to indicate that the data should be read from the request body.

:p How does the `FromBody` attribute work in an ASP.NET Core controller?
??x
The `FromBody` attribute in ASP.NET Core indicates that the data for the action parameter should be read from the request body. This allows complex objects or multiple values to be deserialized directly into model properties.

Code Example:
```csharp
[HttpPost]
public async Task<IActionResult> SaveProduct([FromBody] ProductBindingTarget target)
{
    // Process the valid product object here.
}
```
x??

---

#### Concept of ApiController Attribute in ASP.NET Core Web Services
Background context: This concept explains how the `ApiController` attribute can be used to automatically handle model binding and validation, reducing boilerplate code.

:p What is the purpose of using the `ApiController` attribute in an ASP.NET Core web service?
??x
The `ApiController` attribute simplifies web API development by automatically handling model binding and validation. This means that developers don't need to manually check `ModelState.IsValid`, as it is done implicitly.

Code Example:
```csharp
[ApiController]
public class ProductsController : ControllerBase
{
    // Controller actions here...
}
```
x??

---

#### Concept of Validation Messages in ASP.NET Core Web Services
Background context: This concept discusses how validation errors are communicated back to the client. A `BadRequest` response is typically used, and it includes a JSON object detailing the validation errors.

:p What does the `ModelState` property contain when validation fails?
??x
When validation fails, the `ModelState` property in ASP.NET Core contains information about the errors, including field names and error messages. This data can be accessed to generate a detailed `BadRequest` response.

Code Example:
```csharp
if (!ModelState.IsValid)
{
    return BadRequest(ModelState);
}
```
x??

---

#### Concept of RESTful Web Services with Validation in ASP.NET Core
Background context: This concept involves creating RESTful web services where client data is validated upon reception. If the data is invalid, a `BadRequest` response is returned with specific error details.

:p How can you test validation in an ASP.NET Core web service?
??x
Validation can be tested by sending a POST request to the API endpoint using tools like PowerShell's `Invoke-WebRequest`. A 400 Bad Request response indicates that validation failed, and the body of this response contains JSON detailing the specific validation errors.

Code Example:
```powershell
Invoke-WebRequest http://localhost:5000/api/products -Method POST -Body (@{Name="Boot Laces" | ConvertTo-Json} -ContentType "application/json"
```
x??

---

#### Concept of Error Response in ASP.NET Core Web Services
Background context: This concept explains the structure and content of error responses sent when validation fails. The `BadRequest` response includes a JSON object with field-specific errors.

:p What does the error response body look like when validation fails?
??x
When validation fails, the `BadRequest` response contains a JSON object in the error body that lists each field's validation issues. For example:
```json
{
   "Price":["The field Price must be between 1 and 1000."],
   "CategoryId":["The field CategoryId must be between 1 and 9.223372036854776E+18."],
   "SupplierId":["The field SupplierId must be between 1 and 9.223372036854776E+18."]
}
```
x??

---

#### Removing Null Properties from Web Service Responses
Background context: In this scenario, we are working on improving a web service to avoid sending unnecessary null values in responses. This is particularly important when dealing with navigation properties that are not always populated during simple queries.

:p How can you modify the response from a web service to omit null values?
??x
To remove null values from the response, you have several options:

1. **Explicitly Projecting Selected Properties**: You can manually project the required properties in your controller action and send only those.
2. **Using JSON Serializer Configuration**: Configure the JSON serializer to ignore certain properties when they are `null`.

Here is an example of how to implement both methods:

**Option 1: Explicitly Projecting Selected Properties**

```csharp
[HttpGet("{id}")]
public async Task<IActionResult> GetProduct(long id)
{
    Product? p = await context.Products.FindAsync(id);
    if (p == null)
    {
        return NotFound();
    }
    return Ok(new 
    { 
        p.ProductId, p.Name, p.Price, p.CategoryId, p.SupplierId 
    });
}
```

**Option 2: Using JSON Serializer Configuration**

In the `Product` class:
```csharp
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace WebApp.Models
{
    public class Product
    {
        public long ProductId { get; set; }
        public required string Name { get; set; }
        [Column(TypeName = "decimal(8, 2)")]
        public decimal Price { get; set; }
        public long CategoryId { get; set; }
        public Category? Category { get; set; }
        public long SupplierId { get; set; }
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Supplier? Supplier { get; set; }
    }
}
```

In the `Program.cs` file:
```csharp
using Microsoft.EntityFrameworkCore;
using WebApp.Models;
using Microsoft.AspNetCore.Mvc;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddDbContext<DataContext>(opts =>
{
    opts.UseSqlServer(builder.Configuration["ConnectionStrings:ProductConnection"]);
    opts.EnableSensitiveDataLogging(true);
});

builder.Services.AddControllers();
builder.Services.Configure<JsonOptions>(opts =>
{
    opts.JsonSerializerOptions.DefaultIgnoreCondition
        = JsonIgnoreCondition.WhenWritingNull;
});
```

By applying these methods, you can ensure that the responses from your web service are clean and do not include unnecessary `null` values.
x??

---

#### Using JSONIgnore Condition in Serializer Configuration
Background context: The `JsonIgnore` attribute allows for selective omission of properties during serialization based on specific conditions. This is particularly useful when dealing with nullable navigation properties.

:p How does the `JsonIgnore` attribute help in omitting null values from responses?
??x
The `JsonIgnore` attribute helps in selectively ignoring certain properties during serialization if their value is `null`. By using this attribute, you can ensure that properties which are not relevant or do not have any meaningful data (e.g., navigation properties that might be `null`) are omitted from the JSON response.

Here’s an example of how to apply it:

In your model class:
```csharp
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace WebApp.Models
{
    public class Product
    {
        public long ProductId { get; set; }
        public required string Name { get; set; }
        [Column(TypeName = "decimal(8, 2)")]
        public decimal Price { get; set; }
        public long CategoryId { get; set; }
        public Category? Category { get; set; }
        public long SupplierId { get; set; }
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Supplier? Supplier { get; set; }
    }
}
```

In this example, the `Supplier` property will be ignored in the JSON response if its value is `null`.

Additionally, you can configure the JSON options to default to ignoring properties when they are `null`:
```csharp
builder.Services.Configure<JsonOptions>(opts =>
{
    opts.JsonSerializerOptions.DefaultIgnoreCondition
        = JsonIgnoreCondition.WhenWritingNull;
});
```

By doing this, you ensure that all your responses are clean and do not include unnecessary `null` values.

Example of a response without the `Supplier` property if it's `null`:
```json
{
  "productId": 1,
  "name": "Green Kayak",
  "price": 275.00,
  "categoryId": 1,
  "supplierId": 1
}
```
x??

---

#### Handling Null Values in Complex Queries
Background context: When dealing with navigation properties that are often `null` due to complex queries or lazy loading, it’s important to manage these values properly to avoid unnecessary data transfer and processing.

:p How do you handle null values for navigation properties in a controller action?
??x
Handling null values for navigation properties in a controller action involves selectively including only the necessary data. This can be done by explicitly projecting the required fields or using JSON serialization settings to ignore `null` values.

Here’s an example of handling null values in your controller action:

```csharp
[HttpGet("{id}")]
public async Task<IActionResult> GetProduct(long id)
{
    Product? p = await context.Products.FindAsync(id);
    if (p == null)
    {
        return NotFound();
    }
    // Explicitly project only the required fields
    return Ok(new 
    { 
        p.ProductId, p.Name, p.Price, p.CategoryId, p.SupplierId 
    });
}
```

In this example, you are explicitly selecting and returning only the `ProductId`, `Name`, `Price`, `CategoryId`, and `SupplierId`. This ensures that any related properties like `Category` or `Supplier` will not be included in the response if they are `null`.

Alternatively, you can use JSON serialization settings to ignore null values:

In your model class:
```csharp
using System.ComponentModel.DataAnnotations.Schema;
using System.Text.Json.Serialization;

namespace WebApp.Models
{
    public class Product
    {
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public Supplier? Supplier { get; set; }
    }
}
```

And configure the JSON options in `Program.cs`:
```csharp
builder.Services.Configure<JsonOptions>(opts =>
{
    opts.JsonSerializerOptions.DefaultIgnoreCondition
        = JsonIgnoreCondition.WhenWritingNull;
});
```

This way, if the `Supplier` property is `null`, it will not be included in the JSON response.

Example of a JSON response without the `Supplier` property:
```json
{
  "productId": 1,
  "name": "Green Kayak",
  "price": 275.00,
  "categoryId": 1,
  "supplierId": 1
}
```
x??

---

#### Configuring JSON Serializer in ASP.NET Core
Background context: In ASP.NET Core, the `JsonSerializerOptions` property of the `JsonOptions` class is used to configure how JSON data is serialized. When working with null values, a default policy can be set using the `DefaultIgnoreCondition` property. This affects all JSON responses and should be handled carefully.
:p How does one configure the JSON serializer in ASP.NET Core to manage null values?
??x
To configure the JSON serializer in ASP.NET Core to handle null values, you can use the `JsonSerializerOptions.DefaultIgnoreCondition` property of the `JsonOptions` class. This property determines whether null values should be ignored or included in the JSON output.
```csharp
builder.Services.Configure<JsonOptions>(opts => {
    opts.JsonSerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
});
```
x??

---

#### Rate Limiting Policy Configuration
Background context: Rate limiting is a security mechanism that restricts the number of requests allowed within a specific time frame. In ASP.NET Core, this can be configured using various policies to control access based on IP addresses or other criteria.
:p How does one configure rate limiting in an ASP.NET Core application?
??x
To configure rate limiting in an ASP.NET Core application, you need to set up the rate limiter and apply it to controllers or specific actions. Here's a step-by-step process:

1. **Configure Rate Limiter:**
   ```csharp
   builder.Services.AddRateLimiter(opts => {
       opts.AddFixedWindowLimiter("fixedWindow", fixOpts => {
           fixOpts.PermitLimit = 1;
           fixOpts.QueueLimit = 0;
           fixOpts.Window = TimeSpan.FromSeconds(15);
       });
   });
   ```

2. **Apply Rate Limiting to Controllers:**
   ```csharp
   [EnableRateLimiting("fixedWindow")]
   public class ProductsController : ControllerBase {
       // Controller actions here...
   }
   ```

3. **Disable Rate Limiting for Specific Actions:**
   You can disable rate limiting on specific actions using the `DisableRateLimiting` attribute.
   ```csharp
   [HttpGet("{id}")]
   [DisableRateLimiting]
   public async Task<IActionResult> GetProduct(long id) {
       // Action logic here...
   }
   ```

x??

---

#### Example of Rate Limiting with Controller Actions
Background context: In the provided code, a rate limiting policy is applied to a controller to enforce a specific request limit. This helps prevent abuse and ensures fair usage of API resources.
:p How does one apply rate limiting to specific actions within a controller in ASP.NET Core?
??x
To apply rate limiting to specific actions within a controller in ASP.NET Core, you can use the `EnableRateLimiting` attribute on the controller or individual action methods. Here’s an example:

```csharp
[ApiController]
[Route("api/[controller]")]
[EnableRateLimiting("fixedWindow")] // Applies rate limiting policy 'fixedWindow' to all actions of this controller
public class ProductsController : ControllerBase {
    private DataContext context;

    public ProductsController(DataContext ctx) {
        context = ctx;
    }

    [HttpGet]
    public IAsyncEnumerable<Product> GetProducts() {
        return context.Products.AsAsyncEnumerable();
    }

    [HttpGet("{id}")]
    [DisableRateLimiting] // Disables rate limiting for this specific action
    public async Task<IActionResult> GetProduct(long id) {
        Product? p = await context.Products.FindAsync(id);
        if (p == null) {
            return NotFound();
        }
        return Ok(p);
    }

    // Other actions...
}
```

x??

---

#### Rate Limiting Policy Application in Program.cs
Background context: In the provided configuration, a rate limiting policy is defined and applied to controllers in the `Program.cs` file. This ensures that requests are limited based on predefined criteria.
:p How does one apply a rate limiting policy to all controllers in an ASP.NET Core application?
??x
To apply a rate limiting policy to all controllers in an ASP.NET Core application, you can use the `RequireRateLimiting` method after mapping controllers.

```csharp
app.MapControllers().RequireRateLimiting("fixedWindow");
```

This line ensures that every controller action is subject to the specified rate limiting policy. If you want to apply a different or additional policy to specific actions, you can do so using the `EnableRateLimiting` and `DisableRateLimiting` attributes.

x??

---

