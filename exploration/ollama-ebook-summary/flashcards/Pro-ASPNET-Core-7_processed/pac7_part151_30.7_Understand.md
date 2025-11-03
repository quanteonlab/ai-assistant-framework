# Flashcards: Pro-ASPNET-Core-7_processed (Part 151)

**Starting Chapter:** 30.7 Understanding and changing filter order

---

#### Using Global Filters
Background context: In ASP.NET Core, filters provide a powerful way to intercept and modify the execution flow of an action or result. This is useful for implementing cross-cutting concerns such as logging, authentication, authorization, and error handling.

:p How can global filters be used in ASP.NET Core?
??x
Global filters are applied across all controllers and actions by registering them through `ConfigureServices` method in the `Program.cs` file. They provide a way to enforce common behavior without needing to decorate each controller or action with specific attributes. This is demonstrated by adding a global filter that enforces HTTPS-only policy.

```csharp
builder.Services.Configure<MvcOptions>(opts => {
    opts.Filters.Add<HttpsOnlyAttribute>();
});
```
x??

---

#### Message Attribute Filter
Background context: The `MessageAttribute` class defines a custom result filter that collects multiple messages and displays them to the user. This attribute can be applied at different scopes (controller or action) to build up a series of messages.

:p How does the `MessageAttribute` class work?
??x
The `MessageAttribute` class is an implementation of `IAsyncAlwaysRunResultFilter`, which allows it to run after the result has been executed but before any subsequent filters. It collects multiple messages and stores them in a dictionary that can be accessed by the view.

```csharp
public class MessageAttribute : Attribute, IAsyncAlwaysRunResultFilter {
    private int counter = 0;
    private string msg;

    public MessageAttribute(string message) => msg = message;

    public async Task OnResultExecutionAsync(
        ResultExecutingContext context,
        ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data) 
        { 
            resultData = data; 
        } 
        else 
        { 
            resultData = new Dictionary<string, string>(); 
            context.Result = new ViewResult() 
            { 
                ViewName = "/Views/Shared/Message.cshtml", 
                ViewData = new ViewDataDictionary( 
                    new EmptyModelMetadataProvider(), 
                    new ModelStateDictionary()) 
                { 
                    Model = resultData 
                } 
            }; 
        }

        while (resultData.ContainsKey($"Message_{counter}")) {
            counter++;
        }
        
        resultData[$"Message_{counter}"] = msg;
        await next();
    }
}
```
x??

---

#### Applying Filters to Controllers
Background context: Filters can be applied at different scopes in ASP.NET Core—global, controller-level, or action-level. This flexibility allows developers to manage common behaviors such as logging or authorization without cluttering every method with attributes.

:p How are filters applied to the `HomeController`?
??x
Filters can be applied directly to a specific action or controller by using attributes. In the `HomeController`, multiple instances of the `MessageAttribute` filter are applied at both controller and action levels, allowing for the collection of messages from various sources before rendering the view.

```csharp
[Message("This is the controller-scoped filter")]
public class HomeController : Controller {
    [Message("This is the first action-scoped filter")]
    [Message("This is the second action-scoped filter")]
    public IActionResult Index() {
        return View("Message", "This is the Index action on the Home controller");
    }
}
```
x??

---

#### Global Filters vs. Action/Controller Filters
Background context: Global filters apply to all controllers and actions, while scoped filters (controller or action) apply only to specific scopes. Understanding this distinction helps in managing the application's behavior effectively.

:p What is the difference between global and scoped filters?
??x
Global filters are registered through `ConfigureServices` and apply across all controllers and actions, ensuring uniformity. Scoped filters (like controller or action-level filters) are defined by applying attributes to specific methods or classes, offering more granular control over behavior.

For example:
- Global filter: Enforces HTTPS-only policy globally.
- Controller/action scoped filters: Collect messages from different sources in the `HomeController`.

```csharp
builder.Services.Configure<MvcOptions>(opts => {
    opts.Filters.Add(new MessageAttribute("This is the globally-scoped filter"));
});
```
x??

---

#### Filter Order and Execution Sequence
Background context: Filters execute in a specific order—authorization, resource, action, page, and result. However, if there are multiple filters of the same type, their execution order depends on the scope they are applied to.

:p How do filters with the same type but different scopes execute?
??x
Filters of the same type can have different execution orders based on the scope in which they are applied. For instance, a global filter will run before any scoped filters because it has a broader scope and applies globally. Scopes determine the order, ensuring that more specific filters run after less specific ones.

For example:
- Global filters (run first)
- Controller-scoped filters
- Action-scoped filters

```csharp
// Example of applying multiple filters
[Message("This is the controller-scoped filter")]
public class HomeController : Controller {
    [Message("This is the action-scoped filter 1")]
    public IActionResult Index() {
        return View();
    }
}
```
x??

---

#### Understanding and Changing Filter Order
Background context: In ASP.NET Core, filters are used to add custom behavior before or after a request is processed. The order of filter execution is crucial for implementing logic like authorization checks, modifying response data, etc.

:p How can you change the default filter order in ASP.NET Core?
??x
To change the default filter order, implement the `IOrderedFilter` interface and set the `Order` property to a specific value. Filters with lower `Order` values are applied before those with higher values.
```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class MessageAttribute : Attribute, IAsyncAlwaysRunResultFilter, IOrderedFilter
{
    public int Order { get; set; } // Set the desired order here

    // Filter logic...
}
```
x??

---

#### Applying Filters in Different Scopes
Background context: Filters can be applied at different levels of granularity—global, controller/page model class, and action/handler methods. The default execution order is global filters first, then controllers or page model classes, followed by actions/handlers.

:p What is the default filter order in ASP.NET Core?
??x
The default filter order in ASP.NET Core is as follows:
1. Global Filters: Applied to every request.
2. Controller/Page Model Class Filters: Applied to all actions within a controller/page model class.
3. Action/Handler Method Filters: Applied to individual action methods.

This order ensures that global security checks are performed first, followed by per-controller/page logic and finally specific action-level modifications.
x??

---

#### Implementing the IOrderedFilter Interface
Background context: To change the default filter order, you can implement the `IOrderedFilter` interface. This allows setting a custom `Order` value for your filters.

:p How do you implement the `IOrderedFilter` interface in ASP.NET Core?
??x
To implement the `IOrderedFilter` interface, define a class that implements it and set the `Order` property to control the execution order of the filter. Here’s an example:
```csharp
using Microsoft.AspNetCore.Mvc.Filters;

public class MessageAttribute : Attribute, IAsyncAlwaysRunResultFilter, IOrderedFilter
{
    private int counter = 0;
    private string msg;

    public MessageAttribute(string message) => msg = message;
    
    public int Order { get; set; } // Set the desired order here

    public async Task OnResultExecutionAsync(
        ResultExecutingContext context,
        ResultExecutionDelegate next)
    {
        Dictionary<string, string> resultData;
        if (context.Result is ViewResult vr
            && vr.ViewData.Model is Dictionary<string, string> data)
        {
            resultData = data;
        }
        else
        {
            resultData = new Dictionary<string, string>();
            context.Result = new ViewResult()
            {
                ViewName = "/Views/Shared/Message.cshtml",
                ViewData = new ViewDataDictionary(
                    new EmptyModelMetadataProvider(),
                    new ModelStateDictionary())
                { 
                    Model = resultData 
                }
            };
        }

        while (resultData.ContainsKey($"Message_{counter}"))
        {
            counter++;
        }
        
        resultData[$"Message_{counter}"] = msg;
        await next();
    }
}
```
x??

---

#### Changing the Order in Filters
Background context: You can use constructor arguments to set the `Order` property of filters, ensuring that they are applied before or after other filters.

:p How do you change the order of filters using constructor arguments?
??x
You can change the order by setting the `Order` property through a constructor argument. Here's an example in `HomeController.cs`:
```csharp
using WebApp.Filters;

namespace WebApp.Controllers
{
    [Message("This is the controller-scoped filter", Order = 10)]
    public class HomeController : Controller
    {
        [Message("This is the first action-scoped filter", Order = 1)]
        [Message("This is the second action-scoped filter", Order = -1)]
        public IActionResult Index()
        {
            return View("Message", "This is the Index action on the Home controller");
        }
    }
}
```
Here, `Order` values can be negative to ensure a filter runs before global filters with default order. Restart ASP.NET Core and request `https://localhost:44350` to see the new filter order.
x??

---

#### Summary of Filter Types
Background context: Filters in ASP.NET Core are categorized into six types: authorization, resource, action, page, result, and exception.

:p What are the six types of filters in ASP.NET Core?
??x
The six types of filters in ASP.NET Core are:
1. Authorization Filters: Used to implement a security policy.
2. Resource Filters: Executed before the model-binding process.
3. Action/Handler Method Filters: Executed after the model-binding process.
4. Page Filters: Used for per-page logic (rarely used).
5. Result Filters: Executed before and after a result is used to generate a response.
6. Exception Filters: Executed when an exception is thrown.

These filters allow you to add custom behavior at different stages of request processing, making your application more modular and maintainable.
x??

---

---
#### Preparing for Chapter 31
Background context explaining that this chapter focuses on integrating ASP.NET Core form features with Entity Framework Core to create, read, update, and delete (CRUD) functionality. The objective is to demonstrate how tag helpers, model binding, and model validation work together.

:p What is the primary focus of this chapter?
??x
The primary focus of this chapter is to integrate ASP.NET Core form features with Entity Framework Core to implement CRUD operations on an application, showcasing how tag helpers, model binding, and model validation can be effectively used.
x??

---
#### HomeController Preparation
The chapter provides code to replace the `HomeController.cs` file in the Controllers folder. This setup includes querying data using Entity Framework Core and returning it to a view.

:p What changes are made to the `HomeController.cs` file?
??x
Changes include setting up methods to retrieve categories, suppliers, and products with related data, and initializing the context in the constructor. Additionally, an action method is added that returns a view with product data.
```csharp
private IEnumerable<Category> Categories => context.Categories;
private IEnumerable<Supplier> Suppliers => context.Suppliers;

public HomeController(DataContext data)
{
    context = data;
}

public IActionResult Index()
{
    return View(context.Products.Include(p => p.Category).Include(p => p.Supplier));
}
```
x??

---
#### Index.cshtml File
The `Index.cshtml` file in the Views/Home folder is described. It lists products with details and provides links for actions like edit, delete, and create.

:p What does the `Index.cshtml` file do?
??x
The `Index.cshtml` file displays a table of products, including their ID, name, price, category, and actions (details, edit, delete). It also includes a "Create" button to add new products.
```csharp
@model IEnumerable<Product>

Layout = "_SimpleLayout";

<h4 class="bg-primary text-white text-center p-2">Products</h4>
<table class="table table-sm table-bordered table-striped">
    <thead>
        <tr>
            <th>ID</th>
            <th>Name</th>
            <th>Price</th>
            <th>Category</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
        @foreach (Product p in Model ?? Enumerable.Empty<Product>()) {
            <tr>
                <td>@p.ProductId</td>
                <td>@p.Name</td>
                <td>@p.Price</td>
                <td>@p.Category?.Name</td>
                <td class="text-center">
                    <a asp-action="Details" asp-route-id="@p.ProductId"
                       class="btn btn-sm btn-info">Details</a>
                    <a asp-action="Edit" asp-route-id="@p.ProductId"
                       class="btn btn-sm btn-warning">Edit</a>
                    <a asp-action="Delete" asp-route-id="@p.ProductId"
                       class="btn btn-sm btn-danger">Delete</a>
                </td>
            </tr>
        }
    </tbody>
</table>
<a asp-action="Create" class="btn btn-primary">Create</a>
```
x??

---
#### Product Class Changes
The `Product` class in the Models folder is updated to remove model-level validation and disable remote validation.

:p What changes are made to the `Product.cs` file?
??x
Changes include removing the `[PhraseAndPrice]` attribute, making `Name` a required property with an error message, setting the price range, and using column type for decimal precision. Remote validation attributes are also removed.
```csharp
[Required(ErrorMessage = "Please enter a name")]
public required string Name { get; set; }

[Range(1, 999999, ErrorMessage = "Please enter a positive price")]
[Column(TypeName = "decimal(8, 2)")]
public decimal Price { get; set; }
```
x??

---
#### Program.cs File Updates
The `Program.cs` file disables global filters and adds routes. It also seeds the database with sample data.

:p What are the key changes in the `Program.cs` file?
??x
Key changes include disabling global filters, adding a route that makes it clear when a URL targets a controller, and seeding the database with initial data.
```csharp
// Disabling global filters
builder.Services.AddScoped<GuidResponseAttribute>();

// Defining routes
app.MapDefaultControllerRoute();
app.MapControllerRoute("forms", "controllers/{controller=Home}/{action=Index}/{id?}");

// Seeding database
var context = app.Services.CreateScope().ServiceProvider.GetRequiredService<DataContext>();
SeedData.SeedDatabase(context);
```
x??

---

#### Dropping the Database
Background context: The text explains how to use Entity Framework Core (EF Core) to drop a database associated with an ASP.NET Core Web application. This is necessary when making significant changes that require a fresh database setup.

:p How do you drop the database using EF Core in an ASP.NET Core application?
??x
To drop the database, use the `dotnet ef database drop --force` command. This command drops the database associated with the application's context. The `--force` option ensures that the operation is carried out even if there are no changes to apply.

```powershell
dotnet ef database drop --force
```
x??

---

#### Running the Example Application
Background context: After dropping the database, the example demonstrates how to run an ASP.NET Core Web application using a PowerShell command prompt. The application will display a list of products when accessed via `http://localhost:5000/controllers`.

:p How do you run the example application?
??x
To run the example application, use the following command in a PowerShell command prompt:
```powershell
dotnet run
```
This command starts the ASP.NET Core Web application. Accessing `http://localhost:5000/controllers` will display a list of products.

x??

---

#### Preparing the View Model and the View
Background context: The text explains how to set up a form for CRUD operations on product data using an MVC approach in an ASP.NET Core application. This involves creating a view model class (`ProductViewModel`) that holds both business logic and UI configuration, and a corresponding Razor view.

:p What is the purpose of the `ProductViewModel` class?
??x
The `ProductViewModel` class serves as a container for data and configuration settings used by the MVC form to display product information. It includes properties such as `Product`, which holds the actual product data, and `Action`, `ReadOnly`, `Theme`, and others that control the UI behavior and appearance.

```csharp
namespace WebApp.Models {
    public class ProductViewModel {
        public Product Product { get; set; }
        public string Action { get; set; } = "Create";
        public bool ReadOnly { get; set; } = false;
        public string Theme { get; set; } = "primary";
        public bool ShowAction { get; set; } = true;
        public IEnumerable<Category> Categories { get; set; } = Enumerable.Empty<Category>();
        public IEnumerable<Supplier> Suppliers { get; set; } = Enumerable.Empty<Supplier>();
    }
}
```
x??

---

#### Creating the ProductEditor View
Background context: The text describes how to create a Razor view (`ProductEditor.cshtml`) that uses `ProductViewModel` properties to display and edit product data. This view supports creating, editing, or viewing products depending on the `Action` property.

:p What does the `@model ProductViewModel` declaration in `ProductEditor.cshtml` do?
??x
The `@model ProductViewModel` declaration at the top of the `ProductEditor.cshtml` file specifies that this Razor view is strongly typed to the `ProductViewModel` class. This means that the view has access to all properties and methods defined in `ProductViewModel`, allowing for dynamic data binding and conditional rendering based on the model's state.

```csharp
@model ProductViewModel
```
x??

---

#### Displaying Form Fields Based on ViewModel Properties
Background context: The view uses Razor syntax to conditionally display form fields, buttons, and other UI elements based on properties of the `ProductViewModel` class. This ensures that different forms (create, edit, delete) can share a common template with varying configurations.

:p How does the `readonly` attribute in input fields use the `Model?.ReadOnly` expression?
??x
The `readonly` attribute is conditionally set based on the value of the `ReadOnly` property in the view model. If `ReadOnly` is true, the input field will be read-only; otherwise, it will allow editing.

```html
<input class="form-control" asp-for="Product.Name" readonly="@Model?.ReadOnly" />
```
x??

---

#### Conditional Rendering with Razor Syntax
Background context: The example shows how to use Razor syntax for conditional rendering. For instance, the form submission button is only displayed if `ShowAction` in the view model is true.

:p How does the conditional statement control the visibility of the submit button?
??x
The form submission button is conditionally rendered using a Razor `if` block that checks the value of the `ShowAction` property. If `ShowAction` is true, the button will be displayed; otherwise, it won't appear.

```html
@if (Model?.ShowAction == true) {
    <button class="btn btn-@Model?.Theme mt-2" type="submit">
        @Model?.Action
    </button>
}
```
x??

---

