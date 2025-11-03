# Flashcards: Pro-ASPNET-Core-7_processed (Part 113)

**Starting Chapter:** 21.3 Working with Razor Views

---

#### Razor Views Overview
Background context: Razor views are a combination of HTML and C# expressions used in ASP.NET Core applications to generate dynamic web pages. Expressions are denoted with the `@` character, allowing for dynamic content generation.
:p What are Razor views?
??x
Razor views combine HTML elements and C# expressions, enabling dynamic content generation within an ASP.NET Core application. The @ symbol is used to denote C# expressions that get evaluated during view rendering.
x??

---

#### Compilation of Razor Views
Background context: By default, Razor views are compiled into DLLs as C# classes derived from the `RazorPage<T>` class. These generated classes provide a simplified representation of the view logic for better understanding.
:p How do Razor views get compiled?
??x
Razor views are automatically compiled into C# classes that inherit from `RazorPage<T>`. The generated C# files are typically written to an internal folder during the build process, not directly visible in the project directory by default. To see these generated files, you can enable emitting of compiler-generated files.
x??

---

#### Generated Class Example
Background context: An example of a generated Razor view class is shown, demonstrating how the HTML and C# expressions are transformed into methods within a `RazorPage` derived class.
:p What does the generated Razor view class look like?
??x
The generated class is similar to this:
```csharp
namespace AspNetCoreGeneratedDocument {
    internal sealed class Views_Home_Watersports : RazorPage<dynamic> {
        public async override Task ExecuteAsync() { 
            // Method body with Write and WriteLiteral calls 
        }
        // Various properties representing view-related services
    }
}
```
The `ExecuteAsync` method handles the rendering logic, using methods like `WriteLiteral` for static HTML content and `Write` to include dynamic expressions.
x??

---

#### Caching Responses in Views
Background context: Razor views can cache responses by applying the `ResponseCache` attribute to action methods or controllers. This helps in improving performance by storing frequently accessed page content.
:p How can response caching be enabled in Razor views?
??x
Response caching can be enabled by using the `ResponseCache` attribute on an action method or controller class. For example:
```csharp
[ResponseCache(Duration = 30)]
public IActionResult ActionMethod() {
    // Action logic
}
```
Or for a controller:
```csharp
[ResponseCache(Duration = 30)]
public class MyController : Controller {
    // Controller actions
}
```
This attribute specifies that the response should be cached for the specified duration (in seconds).
x??

---

#### Properties and Methods of RazorPage<T>
Background context: The `RazorPage<T>` base class provides several properties and methods to assist in rendering views, including access to model data and helpers.
:p List some useful properties and methods of `RazorPage<T>`.
??x
Some useful properties and methods of `RazorPage<T>` include:
- **Context**: Returns the current HttpContext object.
- **Model**: Returns the view model passed by the action method.
- **ViewData**: Contains dynamic data passed from the controller to the view.
- **Write(str)**: Writes a string, encoding it for HTML safety.
- **WriteLiteral(str)**: Writes a string without encoding it for use in static regions of HTML.

For example:
```csharp
public override Task ExecuteAsync() {
    WriteLiteral("<p>Hello, @Model.Name!</p>");
}
```
x??

---

---
#### View Properties in Razor Pages
In a Razor Page, the generated view class inherits properties from `RazorPage<T>`, which include additional properties specifically designed for managing different aspects of rendering views. These properties are crucial for handling HTML output, JSON data, and more.

:p What are some key properties defined in the generated view class that help manage different types of content within a Razor Page?
??x
These properties provide various helper methods to assist with encoding, working with URLs, and manipulating the model data directly from the view. For example:
- `Html`: Provides an instance of `IHtmlHelper` for HTML encoding.
- `Json`: Offers functionality to encode data as JSON strings.

Here is a simple example demonstrating how to use these properties in your Razor Page:

```csharp
public class WatersportsModel : PageModel
{
    public string Name { get; set; }
    public decimal Price { get; set; }
    public int CategoryId { get; set; }
    public double TaxRate { get; set; }

    public void OnGet()
    {
        // Set model properties here.
        Name = "Kayak";
        Price = 199.99m;
        CategoryId = 5;
        TaxRate = 0.0825;
    }
}
```

:x?

---
#### Setting ViewModel Type
When creating a Razor Page, the `RazorPage<T>` class requires specifying a generic type argument that represents the view model's data type. However, if not explicitly specified by the developer, Razor defaults to using `dynamic` as the generic type.

:p What happens when you use the default `dynamic` type for the view model in a Razor Page?
??x
Using the `dynamic` type means that properties and methods of the model can be accessed without knowing their exact types at compile time. This can lead to runtime errors if incorrect property names are used, as no static type checking is performed.

For instance, consider this simplified example:

```csharp
@page
@model dynamic

<h1>@Model.Name</h1>
<p>Price: @Model.Price.ToString("c")</p>
```

If `Name` or any other non-existent property like `Brand` is used in the view without being present in the model, an exception will occur at runtime.

:x?

---
#### Handling Runtime Errors with Model Type
To avoid such errors during development, you can explicitly specify the type of the model using the `@model` directive. This allows Visual Studio's Intellisense to provide suggestions for valid properties and methods, reducing the chance of typos and ensuring more robust development.

:p How can you specify the model type in a Razor Page?
??x
You can use the `@model` keyword at the top of your Razor view file to specify the exact type of the model. This tells Visual Studio about the structure of the object, enabling features like Intellisense and compile-time validation.

Example:

```csharp
@page
@model MyProject.Models.Watersport

<h1>@Model.Name</h1>
<p>Price: @Model.Price.ToString("c")</p>
```

In this example, `MyProject` is the namespace of your project, and `Watersport` is the name of your model class.

:x?

---

#### View Model Declaration and Usage
Background context: In Razor Pages, views are strongly typed using view models. This means that when a view references a model, it ensures type safety and provides IntelliSense support during development.

:p What is the purpose of declaring a view model in a Razor Page?
??x
The purpose of declaring a view model in a Razor Page is to ensure type safety and provide IntelliSense support. By specifying the view model as the type parameter for the `RazorPage` class, Visual Studio and Visual Studio Code can offer suggestions for properties and methods defined within that view model while you edit the view.

Example code:
```csharp
internal sealed class Views_Home_Watersports : RazorPage<WebApp.Models.Product>
{
    // Page logic goes here
}
```
x??

---
#### Model Property Updates in Views
Background context: When a view is strongly typed, it can directly reference properties of the model. This means you no longer need to cast or use `Model.` prefix for each property.

:p How should properties be updated when they are not present in the view model?
??x
When properties are not present in the view model, you must add them and ensure the corresponding class (view model) includes these new properties. This ensures that the view can access these properties without errors during development and runtime.

Example code:
```csharp
@model WebApp.Models.Product

<tr>
    <th>Supplier ID</th>
    <td>@Model.SupplierId</td>
</tr>
```
x??

---
#### Visual Studio Editor Suggestions for View Models
Background context: Visual Studio or Visual Studio Code can provide editor suggestions based on the view model's properties and methods. This feature enhances productivity by reducing typos and ensuring that you are using the correct property names.

:p What feature does Visual Studio use to suggest properties and methods in views?
??x
Visual Studio uses IntelliSense to suggest properties and methods from the view model class when editing Razor pages. This helps developers type faster, reduces errors, and ensures they are working with valid member names defined by their view models.

Example of editor suggestions:
- When typing `@Model.` in a Razor file, Visual Studio will show all available properties and methods from the `WebApp.Models.Product` class.
x??

---
#### Correcting Errors in View Models
Background context: If a property is missing in the view model but referenced in the view, it can cause errors. These errors are detected by tools like Visual Studio or through commands such as `dotnet watch`.

:p What happens if a property required by a Razor view is not present in the corresponding view model?
??x
If a property required by a Razor view is not present in the corresponding view model, the system will generate an error. This can be detected immediately in the editor or through command-line tools like `dotnet watch`. The compiler will also report this error when building the project.

Example of an error:
```csharp
<tr>
    <th>Tax Rate</th>
    <td>@Model.TaxRate</td>  // Error: Property 'TaxRate' is not defined in WebApp.Models.Product
</tr>
```
x??

---

#### Adding Namespace to View Imports File
Background context: In Razor views, all types referenced need to be qualified with a namespace by default. However, this can clutter views, especially when using complex expressions. To streamline this process, you can use view imports files which allow you to specify namespaces that should be searched for types.

:p How do you add namespaces to a Razor view without qualifying every type?
??x
You create a file named `_ViewImports.cshtml` in the `Views` folder and then use the `@using` directive to include necessary namespaces. For example:
```cshtml
@using WebApp.Models
```
This allows types from specified namespaces to be used directly in your views without qualifying them with their full namespace path.
x??

---
#### Creating View Imports File Using Visual Studio
Background context: When using Visual Studio, you can easily create a view imports file by right-clicking on the `Views` folder and selecting "Add > New Item," then choosing the appropriate template.

:p How do you add a new _ViewImports.cshtml file in Visual Studio?
??x
Right-click the Views folder in Solution Explorer, select "Add > New Item," and choose the Razor View Imports template from the ASP.NET Core category. This will automatically create an `_ViewImports.cshtml` file that is empty by default.
x??

---
#### Using _ViewImports.cshtml to Simplify Code
Background context: Once you have specified namespaces in `_ViewImports.cshtml`, you can remove these namespaces from your Razor views, making the code more readable and maintainable.

:p What happens when you add `@using WebApp.Models` to an `_ViewImports.cshtml` file?
??x
When you add `@using WebApp.Models` to `_ViewImports.cshtml`, it specifies that the `WebApp.Models` namespace should be searched for types. This means you no longer need to qualify any classes from this namespace with their full namespace path in your Razor views.

For example, if you have a model class named `Product` in `WebApp.Models`, you can use it directly like this:
```cshtml
@model Product
```
x??

---
#### Differentiating View Files and _ViewImports.cshtml
Background context: Files that begin with an underscore (the `_` character) in the Views folder are not returned to the user. Instead, they serve as support files for your views, such as layout files or view imports files.

:p How do you differentiate between a regular Razor view file and a support file like `_ViewImports.cshtml`?
??x
Files in the `Views` folder that start with an underscore are not rendered to the user. For example, `_ViewImports.cshtml` is used internally by your application but will not be served as part of the HTTP response.

You can add such files directly under the Views folder:
- Normal view file: `Home/Watersports.cshtml`
- Support file (not returned to user): `_ViewImports.cshtml`
x??

---
#### Modifying a Razor View After Adding _ViewImports.cshtml
Background context: After adding namespaces in an `_ViewImports.cshtml` file, you can remove the namespace qualifier from your view files. This makes the code more concise and easier to read.

:p What changes should you make to `Watersports.cshtml` after specifying a namespace in `_ViewImports.cshtml`?
??x
After adding `@using WebApp.Models` to `_ViewImports.cshtml`, you can remove the namespace qualifier from your view. For example, if you have the following model class:
```cshtml
@model Product
```
You no longer need to specify `WebApp.Models.Product`. The code becomes more straightforward and readable.

Here is an example of what your modified view might look like:
```cshtml
@model Product

<!DOCTYPE html>
<html>
<head>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
    <h6 class="bg-secondary text-white text-center m-2 p-2">Watersports</h6>
    <div class="m-2">
        <table class="table table-sm table-striped table-bordered">
            <tbody>
                <tr><th>Name</th><td>@Model.Name</td></tr>
                <tr>
                    <th>Price</th>
                    <td>@Model.Price.ToString("c")</td>
                </tr>
                <tr><th>Category ID</th><td>@Model.CategoryId</td></tr>
                <tr><th>Supplier ID</th><td>@Model.SupplierId</td></tr>
            </tbody>
        </table>
    </div>
</body>
</html>
```
x??

---
#### Understanding ViewModel Type Pitfall
Background context: In ASP.NET Core, when using Razor views with controllers, passing an incorrect or unexpected type to the view as a model can lead to runtime errors. This is because the `View` method of the `Controller` class accepts any object, and the compiler does not check for type mismatches at compile time.

:p What happens if a controller passes a wrong model type to the `View` method?
??x
When a controller passes an incorrect or unexpected type as a model to the view, it can lead to runtime errors. The view might expect a specific model type but receive another object type instead. If there is no matching data for that type in the database or if the type contains null values where expected non-null types are required, this mismatch can cause issues during rendering.

For example:
```csharp
public IActionResult WrongModel() {
    return View("Watersports", "Hello, World.");
}
```
This action method passes a string to the `View` method, which expects an object of type `Product`. When such an error occurs, the view may not render correctly or throw runtime exceptions.

x??
---
#### Nullable ViewModel Type Pitfall
Background context: A more subtle issue arises when using nullable types as models in views. While it might seem safe to use a nullable type like `Product?` in your action method, this can lead to run-time errors if the data retrieved from the database is null.

:p How does using a nullable type as a model in a view affect rendering?
??x
Using a nullable type such as `Product?` in your view models can result in runtime errors because Razor views expect non-null types. If the underlying object (e.g., `Product`) is null, attempting to access its properties will cause a runtime exception.

For instance:
```csharp
public async Task<IActionResult> Index(long id = 1) {
    Product? prod = await context.Products.FindAsync(id);
    if (prod?.CategoryId == 1) {
        return View("Watersports", prod);
    } else {
        return View(prod);
    }
}
```
If `prod` is null, the view will try to access properties of a non-existent object, leading to runtime errors.

To handle this issue, you can modify your Razor views to use nullable types or ensure that the action method does not pass null values.
```html
@model Product?
<p>Name: @Model?.Name</p>
```
Here, `@Model?.Name` safely accesses the `Name` property if `Model` is non-null.

x??
---

