# Flashcards: Pro-ASPNET-Core-7_processed (Part 110)

**Starting Chapter:** 20.4.5 Restricting the formats received by an action method

---

---
#### Understanding 406 Not Acceptable Error
Background context: The `Invoke-WebRequest` command returned a 406 Not Acceptable error because there was no overlap between the formats that the client could handle and those that the MVC Framework could produce. This ensures the client does not receive data in a format it cannot process.
:p What causes an HTTP 406 (Not Acceptable) error in this context?
??x
The error occurs when the requested content type is not available from the server. In this case, if the client requests XML and JSON, but the server can only produce JSON for the given action method.
x??

---
#### Specifying Action Result Formats with Produces Attribute
Background context: The `Produces` attribute in ASP.NET Core MVC restricts the data formats that an action method result can use. Multiple formats can be specified to allow flexibility.
:p How does the `Produces` attribute work?
??x
The `Produces` attribute limits the response types based on what the client can accept, as indicated by its Accept header. For example:
```csharp
[HttpGet("object")]
[Produces("application/json")]
public async Task<ProductBindingTarget> GetObject()
{
    Product p = await context.Products.FirstAsync();
    return new ProductBindingTarget() {
        Name = p.Name,
        Price = p.Price,
        CategoryId = p.CategoryId,
        SupplierId = p.SupplierId
    };
}
```
This method restricts the response to JSON, ensuring that if the client requests XML, it will be rejected and the JSON response is used instead.
x??

---
#### Requesting a Format in the URL with FormatFilter Attribute
Background context: The `FormatFilter` attribute allows specifying data formats through the URL, overriding the client's Accept header. This makes the format selection more flexible for clients who cannot control their headers.
:p How does the `FormatFilter` attribute enable requesting different data formats?
??x
The `FormatFilter` attribute in ASP.NET Core MVC allows setting the response type directly via a segment variable in the URL. For example:
```csharp
[HttpGet("object/{format?}")]
[FormatFilter]
[Produces("application/json", "application/xml")]
public async Task<ProductBindingTarget> GetObject()
{
    Product p = await context.Products.FirstAsync();
    return new ProductBindingTarget() {
        Name = p.Name,
        Price = p.Price,
        CategoryId = p.CategoryId,
        SupplierId = p.SupplierId
    };
}
```
This setup means that if a client requests `http://localhost:5000/api/content/object/json`, the response will be in JSON, and similarly for XML.
x??

---
#### Restricting Formats Received by Action Methods with Consumes Attribute
Background context: The `Consumes` attribute restricts what data types an action method can handle. This is useful when ensuring that only certain types of data are processed to avoid deserialization issues or security concerns.
:p How does the `Consumes` attribute work for incoming requests?
??x
The `Consumes` attribute in ASP.NET Core MVC limits the content types that can be sent with an HTTP POST request. For example:
```csharp
[HttpPost]
[Consumes("application/json")]
public string SaveProductJson(ProductBindingTarget product)
{
    return $"JSON: {product.Name}";
}
```
This restricts the method to only process JSON data, ensuring that only requests with `Content-Type` set to `application/json` are handled by this method.
x??

---

#### Sending XML Data Using Invoke-RestMethod
Background context: This example demonstrates how to send XML data using PowerShell's `Invoke-RestMethod`. The request is sent with a specific content type and body containing an XML structure. The action method that handles this request responds with a single piece of data, in this case, the string "Kayak".

:p How does one send XML data via POST request using `Invoke-RestMethod`?
??x
To send XML data, you need to specify the `-Method`, `-Body`, and `-ContentType`. The body should contain the XML payload, and the content type must match what the server expects.
```powershell
Invoke-RestMethod http://localhost:5000/api/content -Method POST -Body "<ProductBindingTarget xmlns=`"http://schemas.datacontract.org/2004/07/WebApp.Models`"> <CategoryId>1</CategoryId><Name>Kayak</Name><Price> 275.00</Price> <SupplierId>1</SupplierId></ProductBindingTarget>" -ContentType "application/xml"
```
x??

---

#### Handling Unsupported Media Type
Background context: When a request's `Content-Type` header doesn't match the application's supported types, it results in a 415 error. This example highlights this situation and explains how to ensure that requests with unsupported content types are handled appropriately.

:p What happens if a request is sent with an unsupported Content-Type?
??x
If a request has a `Content-Type` header that doesn't match the application's supported data types, the application will return a 415 Unsupported Media Type error.
x??

---

#### Configuring Output Caching in ASP.NET Core
Background context: This example demonstrates how to configure output caching in an ASP.NET Core application using the `OutputCache` middleware. By configuring caching policies and applying them to specific actions or controllers, you can control when cached responses are used.

:p How is output caching configured in ASP.NET Core?
??x
Output caching in ASP.NET Core can be configured by setting up a policy with `builder.Services.AddOutputCache`. This policy is then applied to an action or controller using the `OutputCache` attribute. For example:
```csharp
builder.Services.AddOutputCache(opts => {
    opts.AddPolicy("30sec", policy => {
        policy.Cache();
        policy.Expire(TimeSpan.FromSeconds(30));
    });
});
```
Then, apply this caching policy to an action method like so:
```csharp
[HttpGet("string")]
[OutputCache(PolicyName = "30sec")]
public string GetString() => $"DateTime.Now.ToLongTimeString() String response";
```
x??

---

#### Caching Controller Actions in ASP.NET Core
Background context: The `OutputCache` attribute allows fine-grained control over caching responses from controller actions. By applying this attribute to specific action methods, you can cache the output for a specified duration.

:p How does one enable output caching on an individual action method?
??x
To enable output caching on an individual action method in ASP.NET Core, use the `OutputCache` attribute and specify the policy name:
```csharp
[HttpGet("string")]
[OutputCache(PolicyName = "30sec")]
public string GetString() => $"DateTime.Now.ToLongTimeString() String response";
```
This ensures that the response for this action is cached based on the specified caching policy.
x??

---

#### Applying Caching Policies to Controllers
Background context: Output caching can be applied not just at the action method level but also globally or at the controller level. This example shows how to configure and apply a caching policy using attributes.

:p How does one configure and apply an output caching policy to a controller in ASP.NET Core?
??x
To configure and apply an output caching policy to a controller, first define the caching policy:
```csharp
builder.Services.AddOutputCache(opts => {
    opts.AddPolicy("30sec", policy => {
        policy.Cache();
        policy.Expire(TimeSpan.FromSeconds(30));
    });
});
```
Then, apply this policy to a controller action method using the `OutputCache` attribute. For example:
```csharp
[ApiController]
[Route("/api/[controller]")]
public class ContentController : ControllerBase {
    [HttpGet("string")]
    [OutputCache(PolicyName = "30sec")]
    public string GetString() => $"DateTime.Now.ToLongTimeString() String response";
}
```
x??

---
#### OpenAPI Specification Overview
OpenAPI (also known as Swagger) is a standard for describing RESTful web services. It allows developers to understand and interact with web services programmatically through documentation. The description includes details such as the HTTP methods, URL patterns, request bodies, response types, and other metadata.
:p What is OpenAPI, and what does it help with?
??x
OpenAPI helps in documenting RESTful web services so that developers can understand their functionality, parameters, responses, etc., without needing to read through code. It standardizes the way services are described using JSON or YAML files.

Code Example:
```yaml
openapi: 3.0.1
info:
  title: WebApp API
  version: v1
paths:
  /products:
    post:
      summary: Save a product
      operationId: saveProduct
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/Product'
      responses:
        '200':
          description: Product saved successfully
```
x??
---
#### Resolving Action Conflicts in ASP.NET Core Web API
When documenting a web service using OpenAPI, it's essential that each action method has a unique combination of HTTP method and URL pattern. The Consumes attribute must be handled appropriately to ensure no conflicts.
:p How do you resolve action method conflicts when using OpenAPI with ASP.NET Core?
??x
To resolve action method conflicts, ensure each remaining action in the controller has a unique combination of HTTP method and URL. In this case, we remove the separate actions for receiving XML and JSON data by combining them into one HttpPost method.

Code Example:
```csharp
[HttpPost]
[Consumes("application/json")]
public string SaveProductJson(ProductBindingTarget product) {
    return $"JSON: {product.Name}";
}
```
x??
---
#### Installing Swashbuckle Package for OpenAPI Documentation
Swashbuckle is a popular implementation of the OpenAPI specification for ASP.NET Core applications. It automatically generates documentation and includes tools to inspect and test web services.
:p How do you install and configure Swashbuckle in an ASP.NET Core project?
??x
To install Swashbuckle, use the following command:
```shell
dotnet add package Swashbuckle.AspNetCore --version 6.4.0
```
Then, configure Swashbuckle in `Program.cs` as follows:

Code Example:
```csharp
builder.Services.AddSwaggerGen(c => {
    c.SwaggerDoc("v1", new OpenApiInfo { Title = "WebApp", Version = "v1" });
});
app.UseSwagger();
app.UseSwaggerUI(options => {
    options.SwaggerEndpoint("/swagger/v1/swagger.json", "WebApp");
});
```
x??
---
#### Accessing OpenAPI Documentation
After configuring Swashbuckle, you can access the generated documentation through a browser by navigating to `http://localhost:5000/swagger/v1/swagger.json`.
:p How do you access the OpenAPI documentation for an ASP.NET Core application?
??x
You can access the OpenAPI documentation by navigating your web browser to the URL `http://localhost:5000/swagger/v1/swagger.json`. This will provide a JSON representation of the service's API, which includes details about all supported URLs, their HTTP methods, request bodies, and responses.

Example Response:
```json
{
  "openapi": "3.0.1",
  "info": {
    "title": "WebApp",
    "version": "v1"
  },
  "paths": {
    "/products": {
      "post": {
        "summary": "Save a product",
        "operationId": "saveProduct",
        "requestBody": {
          ...
        },
        "responses": {
          ...
        }
      }
    }
  }
}
```
x??
---

