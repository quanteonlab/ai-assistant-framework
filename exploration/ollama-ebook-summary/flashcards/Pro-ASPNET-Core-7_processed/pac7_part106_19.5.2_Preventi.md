# Flashcards: Pro-ASPNET-Core-7_processed (Part 106)

**Starting Chapter:** 19.5.2 Preventing over-binding

---

#### Supporting Cross-Origin Requests (CORS)
Background context explaining CORS and its importance for web services, especially when supporting third-party JavaScript clients. ASP.NET Core provides a built-in service to handle CORS which can be enabled by adding `builder.Services.AddCors();` to the Program.cs file.

:p What is CORS and why is it important in web services?
??x
CORS (Cross-Origin Resource Sharing) allows different origins to make requests to the same server. It helps prevent malicious code from making unauthorized requests on behalf of a user, by performing an initial HTTP request to check that the server will allow requests originating from a specific URL.

```csharp
// Adding CORS service in Program.cs
builder.Services.AddCors();
```
x??

---

#### Using Asynchronous Actions in ASP.NET Core
Explanation about how asynchronous actions can improve performance and scalability of web services by allowing threads to process other requests while waiting for external resources. Mention that this doesn't make the actions faster but increases concurrency.

:p What is the benefit of using asynchronous actions in ASP.NET Core?
??x
Using asynchronous actions benefits web services by increasing the number of concurrent HTTP requests an application can handle without blocking threads during database operations or other I/O-bound tasks. This improves overall performance and scalability, although it doesn't speed up the actual execution time.

```csharp
// Example of an async action in ProductsController.cs
[HttpGet]
public IAsyncEnumerable<Product> GetProducts()
{
    return context.Products.AsAsyncEnumerable();
}

[HttpPost]
public async Task SaveProduct([FromBody] Product product)
{
    await context.Products.AddAsync(product);
    await context.SaveChangesAsync();
}
```
x??

---

#### Preventing Over-Binding in Model Binding
Explanation of over-binding, where model binding processes add unexpected data from the client. The example provided shows an error due to attempting to insert a primary key explicitly.

:p What is over-binding and why is it problematic?
??x
Over-binding occurs when the model binding process adds values that were not intended by the developer. This can lead to issues such as inserting explicit values for identity columns in databases, which are typically assigned automatically. It's crucial to manage data carefully to prevent security vulnerabilities.

```csharp
// Example of over-binding causing an error
[HttpPost]
public async Task SaveProduct([FromBody] Product product)
{
    await context.Products.AddAsync(product);
    await context.SaveChangesAsync();
}
```
x??

---

#### Implementing a Binding Target Class
Explanation on how creating a separate data model class for receiving data can prevent over-binding. The `ProductBindingTarget` class is introduced to handle only specific properties.

:p How does the `ProductBindingTarget` class help in preventing over-binding?
??x
The `ProductBindingTarget` class helps by defining only the necessary properties that the application needs when storing a new object, thus preventing unintended data from being added. The `ToProduct()` method ensures that only the expected properties are used.

```csharp
// ProductBindingTarget.cs file in WebApp/Models folder
namespace WebApp.Models {
    public class ProductBindingTarget {
        public required string Name { get; set; }
        public decimal Price { get; set; }
        public long CategoryId { get; set; }
        public long SupplierId { get; set; }

        public Product ToProduct() => new Product() {
            Name = this.Name,
            Price = this.Price,
            CategoryId = this.CategoryId,
            SupplierId = this.SupplierId
        };
    }
}
```
x??

---

#### Discarding Unwanted Data Values
Background context: When creating RESTful web services using ASP.NET Core, the model binding process can ignore or discard values for read-only properties. This is because model binding typically focuses on matching incoming data with the relevant properties of the model.

:p What happens to read-only property values during the model binding process in ASP.NET Core?
??x
During the model binding process, read-only properties are ignored and their values are discarded. The focus is generally on matching incoming data with writable properties that can be set or updated.
x??

---

#### Action Results in ASP.NET Core
Background context: In ASP.NET Core, action methods can return `IActionResult` objects to control the type of response sent back to the client. This allows for more flexibility and control over HTTP status codes and response contents.

:p How can an action method in ASP.NET Core specify a specific response?
??x
An action method can specify a specific response by returning an object that implements the `IActionResult` interface, known as an action result. This allows the action method to return different types of responses such as 200 OK, 204 No Content, or 400 Bad Request without directly manipulating the `HttpResponse` object.
x??

---

#### Using Ok Action Result
Background context: The `Ok` action result in ASP.NET Core is used to return a 200 OK status code along with optional data. This is commonly used when you want to indicate that the request was successful and include some data as part of the response.

:p How does the `Ok` action result work in ASP.NET Core?
??x
The `Ok` action result returns a 200 OK status code and includes an optional data object in the response body. It is used when you want to indicate that the request was successful and return some data.
```csharp
public IActionResult GetProduct(int id)
{
    var product = _context.Products.FirstOrDefault(p => p.ProductId == id);
    if (product != null)
    {
        return Ok(product);
    }
    else
    {
        return NotFound();
    }
}
```
x??

---

#### Using NoContent Action Result
Background context: The `NoContent` action result is used to indicate that the request was successful but there is no data to be returned. It returns a 204 No Content status code.

:p How does the `NoContent` action result work in ASP.NET Core?
??x
The `NoContent` action result returns a 204 No Content status code, indicating that the request was successful but there is no additional data to return.
```csharp
public IActionResult GetProduct(int id)
{
    var product = _context.Products.FirstOrDefault(p => p.ProductId == id);
    if (product == null)
    {
        return NoContent();
    }
}
```
x??

---

#### Using BadRequest Action Result
Background context: The `BadRequest` action result is used to indicate that the request has a client-side error, such as invalid input or missing required data. It returns a 400 Bad Request status code.

:p How does the `BadRequest` action result work in ASP.NET Core?
??x
The `BadRequest` action result returns a 400 Bad Request status code to indicate that there was an issue with the client's request, such as missing required data or invalid input.
```csharp
public IActionResult SaveProduct(Product product)
{
    if (product.ProductId == null || string.IsNullOrEmpty(product.Name))
    {
        return BadRequest("Invalid input");
    }
    _context.Products.Add(product);
    _context.SaveChanges();
    return Ok(product);
}
```
x??

---

#### Understanding HTTP Status Codes in Web Services
Background context: HTTP status codes provide standardized ways to report the result of an HTTP request. Commonly used statuses include 200 OK for success, 204 No Content when there is no data to send back, and 404 Not Found when a requested resource cannot be found.

:p Why are different HTTP status codes important in web services?
??x
Different HTTP status codes are important because they provide standardized ways to report the result of an HTTP request. They help clients understand whether the request was successful, if there is data available, and if the resource exists or not.
For example:
- 200 OK: The request has succeeded and the response contains the requested data.
- 204 No Content: The server successfully processed the request but does not return any content in the response body.
- 404 Not Found: The server cannot find the resource specified by the URL.
x??

---

---
#### IActionResult Methods Overview
The `IActionResult` methods provide a variety of ways to return responses from an action method. These include returning files, handling errors, performing redirects, and customizing HTTP status codes.

These methods are particularly useful for creating RESTful web services where actions need to handle different scenarios such as returning data, sending files, or redirecting clients to other URLs.

:p What does the `IActionResult` return by default if an action method doesn't explicitly return anything?
??x
If an action method returns null, it is equivalent to returning the result from the NoContent method. This will produce a 204 No Content response.
x??

---
#### Using Ok Method for Successful Responses
The `Ok()` method is used to indicate that the request was successful and to send data back to the client.

:p How does the `GetProduct` action handle when it finds a product?
??x
If the product is found, the `GetProduct` action returns an `Ok(p)` response, which sends the product object as JSON.
```csharp
public async Task<IActionResult> GetProduct(long id) {
    Product? p = await context.Products.FindAsync(id);
    if (p == null) {
        return NotFound();
    }
    return Ok(p);
}
```
x??

---
#### Handling Not Found Responses
The `NotFound()` method is used to indicate that the resource being requested does not exist.

:p How does the `GetProduct` action handle when it doesn't find a product?
??x
If no product is found, the `GetProduct` action returns a 404 Not Found response.
```csharp
public async Task<IActionResult> GetProduct(long id) {
    Product? p = await context.Products.FindAsync(id);
    if (p == null) {
        return NotFound();
    }
    return Ok(p);
}
```
x??

---
#### Using HttpPost for Saving Data
The `HttpPost` attribute is used to indicate that an action method handles HTTP POST requests. The `[FromBody]` parameter specifies that the data should be read from the request body.

:p How does the `SaveProduct` action handle saving a product?
??x
The `SaveProduct` action reads a `ProductBindingTarget` object from the request body, converts it to a `Product`, adds it to the context, and saves changes. If successful, it returns an Ok response with the new product.
```csharp
[HttpPost]
public async Task<IActionResult> SaveProduct([FromBody] ProductBindingTarget target) {
    Product p = target.ToProduct();
    await context.Products.AddAsync(p);
    await context.SaveChangesAsync();
    return Ok(p);
}
```
x??

---
#### Using HttpPut for Updating Data
The `HttpPut` attribute is used to indicate that an action method handles HTTP PUT requests. The data in the request body should be a complete object.

:p How does the `UpdateProduct` action handle updating a product?
??x
The `UpdateProduct` action updates an existing product in the database context, saves changes, and returns an Ok response.
```csharp
[HttpPut]
public async Task UpdateProduct([FromBody] Product product) {
    context.Update(product);
    await context.SaveChangesAsync();
}
```
x??

---
#### Using HttpDelete for Deleting Data
The `HttpDelete` attribute is used to indicate that an action method handles HTTP DELETE requests. The method removes a product from the database based on its ID.

:p How does the `DeleteProduct` action handle deleting a product?
??x
The `DeleteProduct` action creates a new `Product` object with the given ID and empty name, removes it from the context, and saves changes.
```csharp
[HttpDelete("{id}")]
public async Task DeleteProduct(long id) {
    context.Products.Remove(new Product() { 
        ProductId = id, Name = string.Empty });
    await context.SaveChangesAsync();
}
```
x??

---

#### Sending a POST Request Using Invoke-RestMethod
Background context: This concept explains how to send a POST request using PowerShell's `Invoke-RestMethod`. The `-Body` parameter is used to pass a JSON object, and the `-ContentType` parameter specifies the content type as JSON. The example demonstrates creating an instance of a product and sending it to a web API.

:p How do you send a POST request to a web service using PowerShell?
??x
To send a POST request to a web service using PowerShell's `Invoke-RestMethod`, use the following command:
```powershell
Invoke-RestMethod http://localhost:5000/api/products -Method POST -Body @{Name="Boot Laces"; Price=19.99; CategoryId=2; SupplierId=2} | ConvertTo-Json -ContentType "application/json"
```
This command sends a JSON object as the body of the request to the specified URL, and sets the content type to `application/json`. The response is then parsed into an object.
??x
---
#### Redirection in ASP.NET Core Web API
Background context: This concept explains how to implement redirections in an ASP.NET Core web application. The `Redirect` method can be used to perform temporary redirections, while `LocalRedirect` and `LocalRedirectPermanent` throw exceptions if the URL is not local. `RedirectToAction` methods are also mentioned for redirecting to other action methods.

:p How do you implement a temporary redirection in ASP.NET Core using the Redirect method?
??x
To implement a temporary redirection in ASP.NET Core, use the following code:
```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;

namespace WebApp.Controllers
{
    [Route("api/[controller]")]
    public class ProductsController : ControllerBase
    {
        private DataContext context;

        public ProductsController(DataContext ctx)
        {
            context = ctx;
        }

        [HttpGet("redirect")]
        public IActionResult Redirect()
        {
            return Redirect("/api/products/1");
        }
    }
```
The `Redirect` method takes a URL string as an argument and performs a temporary redirection.
??x
---
#### Using RedirectToAction in ASP.NET Core
Background context: This concept explains how to use the `RedirectToAction` method for redirections in ASP.NET Core. The method allows redirecting to another action method within the same controller, providing a more flexible approach compared to using the basic `Redirect` method.

:p How do you redirect to an action method in the same controller using RedirectToAction?
??x
To redirect to an action method in the same controller using `RedirectToAction`, use the following code:
```csharp
using Microsoft.AspNetCore.Mvc;
using WebApp.Models;

namespace WebApp.Controllers
{
    [Route("api/[controller]")]
    public class ProductsController : ControllerBase
    {
        private DataContext context;

        public ProductsController(DataContext ctx)
        {
            context = ctx;
        }

        [HttpGet("redirect")]
        public IActionResult Redirect()
        {
            return RedirectToAction(nameof(GetProduct), new { Id = 1 });
        }
```
The `RedirectToAction` method takes the name of the action method and a route value as arguments. The `nameof` expression is used to specify the action without typos.
??x
---
#### Handling Redirections in ASP.NET Core
Background context: This concept explains how to handle redirections in an ASP.NET Core application, particularly when dealing with local URLs versus potentially malicious user-provided URLs.

:p How do you prevent open redirection attacks in ASP.NET Core?
??x
To prevent open redirection attacks in ASP.NET Core, use the `LocalRedirect` and `LocalRedirectPermanent` methods. These methods throw exceptions if a controller tries to perform a redirection to any URL that is not local.
```csharp
using Microsoft.AspNetCore.Mvc;

namespace WebApp.Controllers
{
    [Route("api/[controller]")]
    public class ProductsController : ControllerBase
    {
        // ... other methods omitted for brevity

        [HttpGet("redirect")]
        public IActionResult Redirect()
        {
            return LocalRedirect("/api/products/1");
        }
```
These methods ensure that only local URLs are used, preventing potential security issues.
??x
---

#### Redirecting to a Route Using RedirectToAction Method
Background context: In ASP.NET Core MVC, `RedirectToAction` and `RedirectToRoute` methods are used for redirection. These methods help in navigating to another action method or URL without performing any processing on the current request. The `RedirectToRoute` method is particularly useful when you need more control over the route used for redirection.

:p How does the `RedirectToRoute` method work, and what are its parameters?
??x
The `RedirectToRoute` method allows redirecting to a specific controller action using route values. It takes an anonymous object as a parameter where keys represent route segments and their corresponding values specify which URL to navigate to. For instance:

```csharp
return RedirectToRoute(new { 
    controller = "Products", 
    action = "GetProduct", 
    Id = 1 
});
```

This will redirect the client to the `GetProduct` action of the `Products` controller with an `Id` parameter set to 1.

x??

---

#### Model Binding and Validation
Background context: When working with form data or JSON payloads in ASP.NET Core, model binding converts incoming HTTP requests into strongly-typed objects. However, this conversion alone does not validate the input data. The MVC framework provides mechanisms to ensure that the required properties are present and meet specific criteria.

:p How can you ensure that a property is required during model binding?
??x
To enforce that a property must be provided by the client, you use the `[Required]` attribute on the corresponding property in your data model class. This tells ASP.NET Core to check if this field has been submitted with a non-null value.

For example:

```csharp
[Required]
public required string Name { get; set; }
```

The `Required` attribute ensures that the `Name` property is mandatory and will throw an error if it's not provided in the request.

x??

---

#### Range Validation for Data
Background context: In addition to ensuring that a field is required, you might want to validate that its value falls within certain limits. The `[Range]` attribute can be used to enforce this constraint on properties of primitive types like `decimal`, `int`, etc.

:p How do you use the `[Range]` attribute for validation?
??x
The `[Range]` attribute is applied to a property to specify a range of acceptable values. It takes two parameters: the minimum and maximum allowed values.

For example:

```csharp
[Range(1, 1000)]
public decimal Price { get; set; }
```

This ensures that the `Price` must be between 1 and 1000 (inclusive).

x??

---

#### Saving a Product with Validation
Background context: Before storing any data in the database, it's crucial to validate the incoming model object. This prevents invalid or null values from being persisted.

:p How does the validation process work before saving a product?
??x
The validation process involves checking if all properties decorated with validation attributes are valid using `ModelState.IsValid`. If the model is valid, the data binding converts the validated model into an entity that can be saved to the database.

Here's how it works in code:

```csharp
[HttpPost]
public async Task<IActionResult> SaveProduct([FromBody] ProductBindingTarget target)
{
    if (ModelState.IsValid)
    {
        Product p = target.ToProduct();
        await context.Products.AddAsync(p);
        await context.SaveChangesAsync();
        return Ok(p);
    }
    return BadRequest(ModelState);
}
```

This method first checks `ModelState.IsValid`, and only proceeds to save the product if all validation constraints are met.

x??

---

