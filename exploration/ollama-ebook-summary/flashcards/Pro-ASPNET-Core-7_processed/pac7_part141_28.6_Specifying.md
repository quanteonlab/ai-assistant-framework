# Flashcards: Pro-ASPNET-Core-7_processed (Part 141)

**Starting Chapter:** 28.6 Specifying a model binding source

---

#### FromForm Attribute Usage
Background context: The `FromForm` attribute is used to override the default model binding process and specify that form data should be used as the source of binding data. This can be useful when you want to ensure that a specific parameter always gets its value from the form data, regardless of where the default search sequence would otherwise look.
:p How does the `FromForm` attribute work in overriding the default model binding process?
??x
The `FromForm` attribute allows you to explicitly specify that the form data should be used as the source for a particular parameter. This overrides the default behavior which typically checks other sources such as routing, query strings, and request bodies.
```csharp
public async Task<IActionResult> Index([FromForm] string userInput)
{
    // The value of 'userInput' will always come from form data,
    // even if it could be found in another source like the query string.
}
```
x??

---

#### Binding to a Collection of Complex Types
Background context: In ASP.NET Core, model binding can handle complex types by default. However, when dealing with collections of complex types (like a list or array), you need to ensure that the names of the input fields are correctly formatted to match the underlying model.
:p How does ASP.NET Core handle binding a collection of `Product` objects?
??x
ASP.NET Core automatically binds collections of complex types by ensuring that each item in the collection has a unique name based on its index. For example, if you have a list of `Product`, the framework will look for input fields named `Products[0].Name`, `Products[1].Name`, etc.
```html
<form method="post">
    <input type="text" name="Products[0].Name" value="Product 1 Name" />
    <input type="text" name="Products[0].Price" value="9.99" />
    <input type="text" name="Products[1].Name" value="Product 2 Name" />
    <input type="text" name="Products[1].Price" value="4.50" />
    <!-- More products -->
    <button type="submit">Submit</button>
</form>
```
x??

---

#### Specifying a Model Binding Source
Background context: The default model binding process in ASP.NET Core looks for data in four places: form values, request body (for web service controllers), routing data, and the query string. However, sometimes you might want to specify that a parameter should be bound from only one of these sources.
:p How can you specify a different source for model binding?
??x
You can use attributes like `FromForm`, `FromRoute`, `FromQuery`, etc., to specify the source of the data for a particular parameter. For example, if you want a specific parameter to always be bound from the query string, you would apply the `FromQuery` attribute.
```csharp
public async Task<IActionResult> Index([FromQuery] long? id)
{
    // The 'id' will come exclusively from the query string.
}
```
x??

---

#### FromBody Attribute for API Controllers
Background context: When working with web service controllers or API endpoints, the default model binding process does not look in the request body. However, if you need to bind data directly from the request body (e.g., JSON payloads), you can use the `FromBody` attribute.
:p How do you specify that a parameter should be bound from the request body?
??x
To bind a parameter directly from the request body, you use the `FromBody` attribute. This is particularly useful for API controllers where data might come in as JSON or another format encoded in the body of the request.
```csharp
[HttpPost]
public IActionResult SubmitForm([FromBody] Product product)
{
    // 'product' will be bound directly from the request body.
}
```
x??

---

#### FromRoute Attribute Example
Background context: The `FromRoute` attribute is used to specify that a parameter should always come from route data. This can be useful when you want to ensure that a specific parameter is consistently bound from the URL, regardless of other sources.
:p How does the `FromRoute` attribute work in ASP.NET Core?
??x
The `FromRoute` attribute ensures that a parameter gets its value exclusively from the routing system. If multiple sources could provide this data (like form values or query strings), using `FromRoute` makes sure that only route data is used.
```csharp
public async Task<IActionResult> Details([FromRoute] long id)
{
    // 'id' will always come from the route data.
}
```
x??

---

#### FromHeader Attribute Usage
Background context: The `FromHeader` attribute allows you to bind a parameter directly from an HTTP header. This can be useful when dealing with APIs where certain settings are passed via headers.
:p How does the `FromHeader` attribute work in model binding?
??x
The `FromHeader` attribute binds a parameter's value from a specific HTTP header. You must provide the name of the header to which you want to bind as a property of the attribute.
```csharp
public async Task<IActionResult> HeaderTest([FromHeader(Name = "X-Custom-Header")] string customValue)
{
    // 'customValue' will be bound from the 'X-Custom-Header' HTTP header.
}
```
x??

---

#### Selecting a Binding Source for Properties
Background context: This concept explains how to bind data sources (query strings, headers) to properties in Razor Pages and controllers. The FromQuery attribute is used to bind property values from query strings.

:p How can you specify that a property should be bound using the query string?
??x
The `FromQuery` attribute is used to indicate that a property's value should come from a query string parameter. For instance, in the provided code snippet:

```csharp
[FromQuery(Name = "Data")]
public Product[] Data { get; set; } = Array.Empty<Product>();
```

This specifies that the array of `Product` objects will be populated with values from the query string named "Data".

x??

---

#### Binding to Headers in HTTP Requests
Background context: This section describes how to bind model properties using headers. The FromHeader attribute is used to specify which header should provide the value for a property.

:p How can you bind a method parameter to an HTTP request header?
??x
The `FromHeader` attribute binds the specified property or parameter's value from a header in the HTTP request. For example:

```csharp
public string Header([FromHeader] string accept) { 
    return $"Header: {accept}"; 
}
```

This method will retrieve the value of the `Accept` header and return it.

x??

---

#### Binding from Multiple Sources
Background context: This explains that different attributes can be used to bind model properties depending on where the data is coming from (query string, request headers). The FromQuery attribute binds values from a query string, while other methods might use headers or cookies.

:p What happens if you want to use both query strings and headers for binding?
??x
You can use multiple attributes like `[FromQuery]` and `[FromHeader]` to bind properties based on different sources. For example:

```csharp
[FromQuery(Name = "Data")]
public Product[] Data { get; set; } = Array.Empty<Product>();

[FromHeader("Custom-Header")]
public string CustomValue { get; set; }
```

Here, `Data` is bound from a query string and `CustomValue` is retrieved from the `Custom-Header` header.

x??

---

#### Binding Complex Types
Background context: This section introduces how complex types can be bound using headers. The FromHeader attribute allows you to bind properties of a model class based on HTTP request headers.

:p How can you bind a complex type (e.g., Product) from an HTTP request header?
??x
To bind a complex type, like `Product`, from a header, you apply the `FromHeader` attribute with the appropriate name:

```csharp
[FromHeader(Name = "Custom-Data")]
public Product[] Data { get; set; } = Array.Empty<Product>();
```

In this example, the `Data` array will be populated with values from the `Custom-Data` header.

x??

---

These flashcards cover the main points of binding properties to different data sources in ASP.NET Core using attributes like `FromQuery` and `FromHeader`.

#### Specifying a Model Binding Source Using Headers
Background context: In ASP.NET Core, model binding is used to convert request data into parameters for action methods. Sometimes, certain header values need to be extracted from HTTP requests and passed as method parameters.

:p How can you bind a value from the `Accept-Language` header in an ASP.NET Core controller action?
??x
You can use the `[FromHeader]` attribute along with specifying the name of the header via the `Name` property. Here's how it works:

```csharp
public string Header([FromHeader(Name = "Accept-Language")] string accept) {
    return $"Header: {accept}";
}
```

In this example, the value from the `Accept-Language` header is bound to the `accept` parameter.

x??

---

#### Using Request Bodies as Binding Sources
Background context: Sometimes, data sent by clients isn't in form data but rather in a request body (e.g., JSON). The `[FromBody]` attribute can be used to bind such data from the request body into method parameters.

:p How would you create an action that binds a `Product` object from the request body?
??x
To handle this, you'd use the `[FromBody]` attribute. Here's how:

```csharp
[HttpPost]
[IgnoreAntiforgeryToken]
public Product Body([FromBody] Product model) {
    return model;
}
```

The JSON content in the request body will be deserialized into a `Product` object and passed to the action method.

x??

---

#### Manual Model Binding
Background context: ASP.NET Core can automatically bind data based on convention, but sometimes you need more control over the binding process. The `TryUpdateModelAsync` method allows for manual model binding.

:p How do you perform manual model binding in an ASP.NET Core application?
??x
You use the `TryUpdateModelAsync<T>` method provided by `PageModel` or `ControllerBase`. Here's how:

```csharp
public class BindingsModel : PageModel {
    public Product Data { get; set; } = new Product() { Name = "Skis", Price = 500 };

    public async Task OnPostAsync([FromForm] bool bind) {
        if (bind) {
            await TryUpdateModelAsync<Product>(Data, "data", p => p.Name, p => p.Price);
        }
    }
}
```

In this example, the `TryUpdateModelAsync` method is used to bind data only when the checkbox is checked.

x??

---

