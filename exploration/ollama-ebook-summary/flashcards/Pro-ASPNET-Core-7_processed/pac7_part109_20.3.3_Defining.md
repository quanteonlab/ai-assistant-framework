# Flashcards: Pro-ASPNET-Core-7_processed (Part 109)

**Starting Chapter:** 20.3.3 Defining the action method

---

#### Advanced Web Service Features for Suppliers
Background context: This section explains advanced features of web services, specifically focusing on how to handle supplier and product data. It discusses techniques like breaking circular references using `foreach` loops and handling edit operations with HTTP PATCH requests.

:p What does the provided code do in terms of handling supplier and product data?
??x
The code retrieves a supplier by their ID along with related products. If there are products, it sets each product's Supplier property to null to break circular references before returning the supplier. This is useful when serializing data for consumption.
```csharp
Supplier supplier = await context.Suppliers
    .Include(s => s.Products)
    .FirstAsync(s => s.SupplierId == id);
if (supplier.Products != null) {
    foreach (Product p in supplier.Products) {
        p.Supplier = null;
    };
}
return supplier;
```
x??

---
#### Handling HTTP PATCH Requests for Simple Data Types
Background context: This section explains using the PUT method for simple data types but highlights that for more complex scenarios, a PATCH request is needed. A PATCH request allows sending only the changes to the web service rather than a complete replacement object.

:p How does the provided example handle changing a supplier's name and city using JSON Patch?
??x
The example uses a JSON Patch document with the `replace` operation to change specific properties of a supplier, such as its Name and City. This method is applied through an HTTP PATCH request.
```json
[
    { "op": "replace", "path": "/Name", "value": "Surf Co" },
    { "op": "replace", "path": "/City", "value": "Los Angeles" }
]
```
x??

---
#### Installing and Configuring JSON Patch Package
Background context: This section explains how to support the JSON Patch standard in an ASP.NET Core application by installing and configuring the necessary package. The `JsonPatchDocument` class is used to apply changes to objects.

:p How does one enable the Newtonsoft.Json serializer for handling JSON Patch documents?
??x
To enable the Newtonsoft.Json serializer, you add the `AddNewtonsoftJson()` method to your `Program.cs` file. This configures the application to use the Newtonsoft.Json library instead of the default ASP.NET Core serializer.
```csharp
builder.Services.AddControllers().AddNewtonsoftJson();
```
x??

---
#### Defining Action Method for HTTP PATCH Requests
Background context: The provided code defines an action method in the `SuppliersController` that handles HTTP PATCH requests. It uses a `JsonPatchDocument<Supplier>` to apply changes to a supplier object.

:p How does the `PatchSupplier` action method handle the JSON Patch document?
??x
The `PatchSupplier` action method retrieves a supplier by ID, applies the operations in the provided JSON Patch document using the `ApplyTo` method of `JsonPatchDocument`, and saves any changes to the database.
```csharp
[HttpPatch("{id}")]
public async Task<Supplier?> PatchSupplier(long id,
    JsonPatchDocument<Supplier> patchDoc) {
    Supplier? s = await context.Suppliers.FindAsync(id);
    if (s != null) {
        patchDoc.ApplyTo(s);
        await context.SaveChangesAsync();
    }
    return s;
}
```
x??

---

---
#### Understanding Content Formatting in Web Services
Content formatting is crucial for web services to ensure that data is sent and received correctly. The content format selected depends on several factors, including client preferences, application capabilities, action method policies, and the type of data returned.

:p How does ASP.NET Core determine the content format when no restrictions are applied?
??x
ASP.NET Core uses a default policy for determining the content format based on the return type of an action method. If the action method returns a string, it is sent unmodified with `Content-Type: text/plain`. For all other data types, including simple types like int, the data is formatted as JSON and `Content-Type` is set to `application/json`.

```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;

namespace WebApp.Controllers {
    [ApiController]
    [Route("/api/[controller]")]
    public class ContentController : ControllerBase {
        private DataContext context;
        
        public ContentController(DataContext dataContext) {
            context = dataContext;
        }
        
        [HttpGet("string")]
        public string GetString() => "This is a string response";
        
        [HttpGet("object")]
        public async Task<Product> GetObject() {
            return await context.Products.FirstAsync();
        }
    }
}
```

x??
---
#### Example of Default Content Policy with String Return Type
The default policy treats strings specially. When an action method returns a string, it is sent as `text/plain` to avoid double-encoding issues.

:p What happens when the `GetString` action method returns a string in this example?
??x
When the `GetString` action method returns a string, ASP.NET Core sends the string unmodified with the `Content-Type: text/plain` header. This avoids double encoding issues that might arise if strings were encoded as JSON.

```powershell
Invoke-WebRequest http://localhost:5000/api/content/string | select @{n='Content-Type';e={ $_.Headers."Content-Type" }}, Content
```

The output will show `Content-Type: text/plain; charset=utf-8` and the string response "This is a string response".

x??
---
#### Example of Default Content Policy with Object Return Type
For non-string data types, such as objects or complex types, ASP.NET Core formats the data as JSON and sets the `Content-Type` header to `application/json`.

:p What happens when the `GetObject` action method returns an object in this example?
??x
When the `GetObject` action method returns an object (in this case, a `Product`), ASP.NET Core serializes it into JSON format and sends it with the `Content-Type: application/json` header. This ensures that the data is properly formatted for consumption by clients expecting JSON.

```powershell
Invoke-WebRequest http://localhost:5000/api/content/object | select @{n='Content-Type';e={ $_.Headers."Content-Type" }}, Content
```

The output will show `Content-Type: application/json; charset=utf-8` and the serialized product object in JSON format.

x??
---

#### Content Negotiation Basics
Content negotiation allows a client to specify acceptable response formats for a request. The `Accept` header in HTTP requests is used to indicate these preferences, typically represented as MIME types.

:p What does the `Accept` header allow clients to do?
??x
The `Accept` header enables clients like web browsers and applications to communicate their preferred content formats to the server. This way, servers can return responses in a format that best suits the client's needs.
x??

---

#### Example of Accept Header from Google Chrome
Here’s an example of what an `Accept` header might look like for Google Chrome:

```
Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif, image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
```

:p What does this `Accept` header tell us about Google Chrome's preferences?
??x
This `Accept` header tells us that:
- Google Chrome prefers to receive HTML or XHTML content.
- It also supports XML, AVIF images, WebP images, and APNG images with a higher preference than other formats (q=0.8).
- The format `*/*;q=0.8` indicates that the client will accept any format but ranks it as less preferred compared to others specified.

The q values represent relative preferences where 1.0 is the highest, and lower numbers indicate lesser preference.
x??

---

#### ASP.NET Core Handling of Accept Header
When an `Accept` header is set in a request, such as setting it to `application/xml`, ASP.NET Core may not return the requested format unless configured or otherwise instructed.

:p How did the response change when the `Accept` header was set to `application/xml`?
??x
Setting the `Accept` header to `application/xml` in this case did not result in a JSON response as expected. Instead, the ASP.NET Core application returned a JSON response regardless of the `Accept` header provided.

The command:
```powershell
Invoke-WebRequest http://localhost:5000/api/content/object -Headers @{Accept="application/xml"} | select @{n='Content-Type';e={$_.Headers."Content-Type"}}, Content
```

Resulted in a JSON response being returned:

```
Content-Type                    Content
------------                    -------
application/json; charset=utf-8 {"productId":1,"name":"Kayak","price":275.00,"categoryId":1,"supplierId":1}
```

This indicates that the ASP.NET Core application did not honor the `Accept` header for content negotiation in this scenario.
x??

---

#### Understanding q Values
The q values in an `Accept` header specify relative preference, where 1.0 is the highest and lower numbers indicate lesser preference.

:p What do the q values in the `Accept` header mean?
??x
Q values in the `Accept` header indicate the relative preference for each format:

- Higher q value (e.g., 1.0) means higher preference.
- Lower q value (e.g., 0.9, 0.8) indicates lesser preference.

For example, in the header:
```
text/html,application/xhtml+xml,application/xml;q=0.9,image/avif, image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9
```

- HTML and XHTML have the highest preference (1.0).
- XML has a higher preference than other image formats but lower than HTML or XHTML.
- The `*/*` format is the least preferred, with a q value of 0.8.

The server can use these values to choose the best available response format based on client preferences.
x??

---

#### Content Negotiation and Format Support
Background context: The provided text discusses how to enable content negotiation for different data formats, specifically XML, within an ASP.NET Core MVC framework. This is crucial for applications that need to support various client expectations regarding response formats.

:p How can you enable XML formatting in your ASP.NET Core MVC application?
??x
To enable XML formatting, you need to configure the `AddControllers` service by adding the `AddXmlDataContractSerializerFormatters()` method. This step ensures that your application supports both JSON and XML as output formats.
```csharp
builder.Services.AddControllers()
    .AddNewtonsoftJson() // For JSON support
    .AddXmlDataContractSerializerFormatters(); // Enable XML support
```
x??

---

#### Custom Data Format Implementation
Background context: The text mentions that while custom data format implementations are possible, they are rarely used because the most common formats (JSON and XML) are already implemented. This is useful for understanding the flexibility of the framework but not necessarily for practical implementation.

:p Why might someone choose to implement a custom data format in an ASP.NET Core application?
??x
Implementing a custom data format in an ASP.NET Core application is rarely necessary because JSON and XML are well-supported by default through `AddNewtonsoftJson()` and `AddXmlDataContractSerializerFormatters()`. However, understanding how to create one can be useful for educational purposes or specialized scenarios where neither of these formats meet specific requirements. You would derive from the `OutputFormatter` class to create a custom formatter.
```csharp
public class CustomFormatter : OutputFormatter
{
    // Implement logic here
}
```
x??

---

#### Entity Framework Core Navigation Properties and Serialization Limitations
Background context: The text highlights that XML serialization has limitations, especially with Entity Framework Core navigation properties. It explains why direct entity serialization might not be ideal due to the nature of how these properties are defined.

:p What limitation does XML serialization face when dealing with Entity Framework Core entities?
??x
XML serialization faces a limitation where it cannot handle Entity Framework Core navigation properties because they are typically defined through interfaces rather than concrete classes. To work around this, you may need to create an object that contains the necessary data for serialization, as demonstrated in Listing 20.15.
```csharp
public class ProductBindingTarget
{
    public string Name { get; set; }
    public decimal Price { get; set; }
    public int CategoryId { get; set; }
    public int SupplierId { get; set; }
}
```
x??

---

#### ContentController and Data Binding
Background context: The text provides an example of how to create a controller in ASP.NET Core that handles requests for different data formats, including XML.

:p How does the `ContentController` handle GET requests for both string responses and object serialization?
??x
The `ContentController` uses attributes to define its actions. For a simple string response, it directly returns a string. For more complex objects, it creates an instance of the required target class and populates it with data from the database.
```csharp
[HttpGet("string")]
public string GetString() => "This is a string response";

[HttpGet("object")]
public async Task<ProductBindingTarget> GetObject()
{
    Product p = await context.Products.FirstAsync();
    return new ProductBindingTarget
    {
        Name = p.Name,
        Price = p.Price,
        CategoryId = p.CategoryId,
        SupplierId = p.SupplierId
    };
}
```
x??

---

#### Rate Limiting Configuration
Background context: The text includes configuration for rate limiting, which helps control how many requests a client can make in a given period.

:p How is the rate limiter configured to allow only one request every 15 seconds?
??x
The rate limiter is configured using the `AddFixedWindowLimiter` method. This method sets up a fixed window where only one permit is allowed within the specified time frame.
```csharp
builder.Services.AddRateLimiter(opts =>
{
    opts.AddFixedWindowLimiter("fixedWindow", fixOpts =>
    {
        fixOpts.PermitLimit = 1;
        fixOpts.QueueLimit = 0;
        fixOpts.Window = TimeSpan.FromSeconds(15);
    });
});
```
x??

---

#### Content Negotiation Process in ASP.NET Core MVC
Background context explaining how content negotiation works in ASP.NET Core MVC. The framework respects `Accept` headers to determine the desired response format, but has some odd behaviors that can cause confusion.

:p What is the content negotiation process in ASP.NET Core MVC?
??x
The content negotiation process allows the client to specify the preferred media type of the response using the `Accept` header. If no specific format is requested, or if a fallback format is allowed, the framework may default to JSON even when XML is preferred.

Example code:
```csharp
Invoke-WebRequest http://localhost:5000/api/content/object -Headers @{Accept="application/xml"} | select @{n='Content-Type'; e={ $_.Headers."Content-Type" }}, Content
```
x??

---

#### Fallback Behavior in Content Negotiation
Explanation of how the `*/*` fallback in the `Accept` header can cause JSON responses even when a specific XML format is requested.

:p How does the `*/*` fallback in the `Accept` header affect content negotiation?
??x
The `*/*` fallback in the `Accept` header tells the MVC Framework to accept any format, which causes it to return a JSON response instead of honoring the preference for XML. This can be confusing as the client's preferred format is not respected.

Example code:
```powershell
Invoke-WebRequest http://localhost:5000/api/content/object -Headers @{Accept="application/xml,*/*;q=0.8"} | select @{n='Content-Type'; e={ $_.Headers."Content-Type" }}, Content
```
x??

---

#### Handling Unsupported Formats in Content Negotiation
Explanation of how the MVC Framework handles requests for unsupported formats and sends JSON responses by default.

:p How does ASP.NET Core MVC handle requests for unsupported data formats?
??x
When a client requests an unsupported format, such as `img/png`, the MVC Framework defaults to sending a JSON response. This can be seen when running commands that request an unsupported content type, resulting in a JSON response with a 406 status code.

Example code:
```powershell
Invoke-WebRequest http://localhost:5000/api/content/object -Headers @{Accept="img/png"} | select @{n='Content-Type'; e={ $_.Headers."Content-Type" }}, Content
```
x??

---

#### Configuring Respect for `Accept` Headers in ASP.NET Core MVC
Explanation of how to configure the MVC Framework to respect client preferences specified in the `Accept` header and avoid defaulting to JSON.

:p How can you configure ASP.NET Core MVC to respect client preferences in the `Accept` header?
??x
To configure the MVC Framework to respect client preferences, add settings to the `Program.cs` file. This involves setting `RespectBrowserAcceptHeader` to true and `ReturnHttpNotAcceptable` to true.

Example configuration:
```csharp
builder.Services.Configure<MvcOptions>(opts =>
{
    opts.RespectBrowserAcceptHeader = true;
    opts.ReturnHttpNotAcceptable = true;
});
```

x??

---

