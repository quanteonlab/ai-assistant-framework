# Flashcards: Pro-ASPNET-Core-7_processed (Part 138)

**Starting Chapter:** 28.3.1 Binding simple data types in Razor Pages

---

---
#### Binding Simple Data Types
Background context: In ASP.NET Core, model binding allows simple data types from HTTP requests to be automatically converted into C# values. This makes it easier to handle form inputs and other request parameters without manually parsing them.

:p How does model binding work for simple data types in an ASP.NET Core application?
??x
Model binding in ASP.NET Core uses a default model binder that converts the incoming request data (like form fields) into C# objects or primitive types. This process simplifies handling HTTP requests by automatically converting string values from forms to appropriate .NET types such as `decimal`, `bool`, and `DateTime`.

For example, consider an input in a form:
```html
<input type="text" name="price" value="49.99">
```
The model binder will convert the string "49.99" into a decimal value when processing the request.

In the provided code snippet, the `SubmitForm` action method uses parameters to bind simple data types directly from the request:
```csharp
[HttpPost]
public IActionResult SubmitForm(string name, decimal price) {
    TempData["name param"] = name;
    TempData["price param"] = price.ToString();
    return RedirectToAction(nameof(Results));
}
```
x??

---
#### Example of Using Model Binding in Controllers
Background context: The `SubmitForm` action method demonstrates how to use model binding with simple data types. It binds the values from the request into parameters directly, simplifying the action method and ensuring that string input is converted to a decimal value.

:p How does the `SubmitForm` method utilize model binding?
??x
The `SubmitForm` method uses model binding by accepting parameters of type `string` for `name` and `decimal` for `price`. The model binder automatically converts these values from the request data into their corresponding C# types. For instance, a form input like:
```html
<input type="text" name="name" value="Product X">
```
would be bound to the `name` parameter as a string.

Here is how it looks in the method definition:
```csharp
[HttpPost]
public IActionResult SubmitForm(string name, decimal price) {
    // The model binder converts 'name' and 'price' from form inputs into these parameters.
    TempData["name param"] = name;  // Store the name in temp data as a string.
    TempData["price param"] = price.ToString();  // Convert the price to a string before storing it.
    return RedirectToAction(nameof(Results));  // Redirects to the 'Results' action.
}
```
x??

---
#### Binding Simple Data Types in Razor Pages
Background context: In Razor Pages, model binding can also be used for simple data types. However, care must be taken when using form helpers like `asp-for` because it may change the name attribute of input elements, which could mismatch with the handler method parameters.

:p How does model binding work in Razor Pages for simple data types?
??x
In Razor Pages, model binding works similarly to controllers but requires explicit naming if form fields use helper attributes that might alter their names. For instance, consider a product named `Product`:
```html
<div class="form-group">
    <label>Name</label>
    <input class="form-control" asp-for="Product.Name" name="name" />
</div>

<div class="form-group">
    <label>Price</label>
    <input class="form-control" asp-for="Product.Price" name="price" />
</div>
```
The `asp-for` tag helper sets the names of the input fields to `Product.Name` and `Product.Price`. If these names do not match the handler method parameters, model binding will fail. Therefore, it is necessary to explicitly define the names as shown above.

In the handler method:
```csharp
public class FormHandlerModel : PageModel {
    public IActionResult OnPost(string name, decimal price) {
        // The model binder binds 'name' and 'price' from form inputs.
        TempData["name param"] = name;  // Store the name in temp data as a string.
        TempData["price param"] = price.ToString();  // Convert the price to a string before storing it.
        return RedirectToPage("FormResults");
    }
}
```
x??

---

#### Model Binding and Default Values
Model binding is a feature in ASP.NET Core where data from HTTP requests (such as form submissions) is automatically bound to action method parameters. However, if no value can be found for a parameter, the model binder uses default values. For `long` arguments, this default value is 0.
:p How does model binding handle missing values for `long` parameters?
??x
When a `long` parameter has no available value for model binding, it defaults to 0. This can cause issues if your application expects non-zero values or when querying the database with zero as a key might not return valid results.
```csharp
public async Task<IActionResult> Index(long id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products
        .Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstAsync(p => p.ProductId == id) ?? new() { Name = string.Empty });
}
```
x??

---

#### Handling Missing Values with `FirstOrDefaultAsync`
To handle the issue of missing values, you can use `FirstOrDefaultAsync` instead of `FirstAsync`. This method returns null if no matching object is found in the database, preventing errors.
:p How does using `FirstOrDefaultAsync` help handle missing values?
??x
Using `FirstOrDefaultAsync` ensures that your application doesn't throw an error when no value is available for model binding. It safely handles cases where there might be no matching record in the database by returning null, which can then be checked and handled appropriately.
```csharp
public async Task<IActionResult> Index(long id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products
        .Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => p.ProductId == id) ?? new() { Name = string.Empty });
}
```
x??

---

#### Nullable Parameter for Value Differentiation
If your application needs to differentiate between a missing value and any value provided by the user, you can use a nullable parameter type (`long?`). This allows the parameter to be null if no suitable value is present.
:p How does using a nullable parameter help in differentiating between missing values and user-provided values?
??x
Using a nullable parameter (e.g., `long?`) helps your application distinguish between cases where there is no valid input from users and situations where the input is explicitly set to null. This can be particularly useful for handling optional or non-mandatory fields.
```csharp
public async Task<IActionResult> Index(long? id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products.Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => id == null || p.ProductId == id));
}
```
x??

---

---
#### Binding Complex Types to Database Objects
Background context: The model binding system can handle complex types, which are typically objects with multiple properties. In this scenario, a `Product` object is being bound from request data and used to interact with a database.

:p How does the model binding system work when binding a `Product` object in ASP.NET Core?
??x
The model binding process inspects the `Product` object's public properties and attempts to bind values for each property from the incoming HTTP request. If a value is not present, it uses default values (such as zero for integers or null for reference types).

```csharp
public IActionResult Index(long? id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products
        .Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => id == null || p.ProductId == id) ??
        new() { Name = string.Empty });
}
```
x??

---
#### Binding Complex Types from Request Data to a Form
Background context: In the `FormController`, an `Index` action method is set up to handle both GET and POST requests. The GET request uses model binding to populate a form with existing data, while the POST request processes submitted form data.

:p How does the `Index` action method use model binding for complex types?
??x
The `Index` method fetches a `Product` object from the database based on an optional `id`. If no `id` is provided (i.e., `id == null`), it retrieves the first product. Otherwise, it selects the product with the matching `ProductId`.

```csharp
public async Task<IActionResult> Index(long? id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products
        .Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => id == null || p.ProductId == id) ??
        new() { Name = string.Empty });
}
```
x??

---
#### Submitting Form Data and Handling Missing Values
Background context: The `SubmitForm` action method uses model binding to bind form data to a `Product` object. If any property is missing, it retains the default value.

:p How does the `SubmitForm` method handle properties that have no submitted values?
??x
The `SubmitForm` method creates a new `Product` object and passes it to the action method. The model binding system attempts to bind each public property of the `Product` object from the request data. If a value is not found, the property retains its default value.

```csharp
[HttpPost]
public IActionResult SubmitForm(Product product)
{
    TempData["product"] = JsonSerializer.Serialize(product);
    return RedirectToAction(nameof(Results));
}
```
x??

---
#### Querying Products Based on ID or Default
Background context: The `Index` method uses a query to find a specific product based on the provided `id`. If no `id` is given, it fetches the first available product.

:p How does the `Index` action method determine which product to display?
??x
The `Index` method checks if an `id` was provided. If not (`id == null`), it retrieves the first product from the database using `FirstOrDefaultAsync`. Otherwise, it looks for a specific product by its `ProductId`.

```csharp
public async Task<IActionResult> Index(long? id)
{
    ViewBag.Categories = new SelectList(context.Categories, "CategoryId", "Name");
    return View("Form", await context.Products
        .Include(p => p.Category)
        .Include(p => p.Supplier)
        .FirstOrDefaultAsync(p => id == null || p.ProductId == id) ??
        new() { Name = string.Empty });
}
```
x??

---

