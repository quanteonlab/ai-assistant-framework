# Flashcards: Pro-ASPNET-Core-7_processed (Part 112)

**Starting Chapter:** 21.2.2 Creating an HTML controller

---

---
#### Configuring ASP.NET Core for HTML Responses
Background context: This section explains how to configure an ASP.NET Core application to support HTML responses. The AddControllersWithViews method is used to enable view controllers, which are different from web service controllers.

:p How do you enable support for views in an ASP.NET Core application?
??x
To enable support for views, the `AddControllersWithViews()` method is called on the services collection. This registers both controller and view components with the application.
```csharp
builder.Services.AddControllersWithViews();
```
x??

---
#### HTML Controller Similarities and Differences
Background context: HTML controllers are similar to web service controllers but have important differences, such as using `Controller` instead of `ApiController`. HTML controllers rely on convention-based routing.

:p How do HTML controllers differ from web service controllers?
??x
HTML controllers use the `Controller` class derived from `ControllerBase`, which provides methods for working with views. They do not require `ApiController` attributes and typically use convention-based routing, whereas web service controllers might have specific route attributes applied.
x??

---
#### Convention-Based Routing in HTML Controllers
Background context: HTML controllers rely on convention-based routing rather than explicit attribute routing. The default URL conventions map the controller and action method names to URLs.

:p How does convention-based routing work for HTML controllers?
??x
Convention-based routing uses the controller class name (without "Controller" suffix) as the first segment of the URL, followed by the action method name as the second segment. An optional third segment can be an ID parameter. The default route maps to the Index action on the Home controller if no segments are present.
```csharp
app.MapDefaultControllerRoute();
```
x??

---
#### Configuring ASP.NET Core with Static Files
Background context: To serve static files like HTML, CSS, and JavaScript, you need to configure your application using `UseStaticFiles()`.

:p How do you enable serving of static files in an ASP.NET Core app?
??x
To enable serving of static files, use the `app.UseStaticFiles();` method. This allows your application to serve static content from a directory specified in the file system.
```csharp
app.UseStaticFiles();
```
x??

---
#### Using ViewResult with MVC Framework
Background context: The `View()` method in controllers returns an `IActionResult`, specifically of type `ViewResult`. It tells the MVC framework that a view should be used to produce the response.

:p How does the `View()` method work?
??x
The `View()` method selects a view based on the provided model and returns it as an `ViewResult` which implements `IActionResult`. The method name (action) and parameter are mapped to a specific view.
```csharp
return View(await context.Products.FindAsync(id));
```
x??

---

#### Understanding MVC Action Methods and HTTP Methods
Background context: In ASP.NET Core, an action method is a public method defined within an HTML controller that can handle different types of HTTP requests. By default, any public method without specific attributes is considered an action method supporting all HTTP methods. However, you can restrict which HTTP methods the action method supports by applying specific attributes.

:p What are the attributes used to denote specific HTTP methods in MVC?
??x
The `HttpGet`, `HttpPost`, and other related attributes are used to denote specific HTTP methods that an action method should handle.
```csharp
[HttpGet]
public IActionResult Index(long id = 1)
{
    return View(await context.Products.FindAsync(id));
}

[HttpPost]
public IActionResult Save(Product product)
{
    // save logic here
}
```
x??

---

#### Action Method with Default Parameter
Background context: An action method can have default parameter values, allowing you to define a method that behaves differently based on the presence or absence of parameters.

:p How does an action method handle default parameters in ASP.NET Core?
??x
An action method can define default parameter values. When no value is provided for the parameter, it uses the default value defined in the method signature.
```csharp
public async Task<IActionResult> Index(long id = 1)
{
    return View(await context.Products.FindAsync(id));
}
```
x??

---

#### Routing and Action Methods
Background context: The routing mechanism in ASP.NET Core allows you to define multiple routes, each mapping to a specific action method. By default, the MVC Framework assumes any public method within an HTML controller is an action method.

:p How does the MVC Framework map requests to action methods?
??x
The MVC Framework maps requests to action methods based on the route defined and the name of the controller and action method.
```csharp
public async Task<IActionResult> Index(long id = 1)
{
    return View(await context.Products.FindAsync(id));
}
```
x??

---

#### Razor View Naming Conventions
Background context: When an action method invokes the `View` method, it creates a `ViewResult`, which tells the MVC Framework to use the default convention to locate a view. The view name is derived from the action method and controller names.

:p How does the Razor view engine determine the location of a view?
??x
The Razor view engine searches for views in the following order:
1. `Views/{Controller}/{Action}.cshtml` (e.g., `Views/Home/Index.cshtml`)
2. `Views/Shared/{Action}.cshtml`
```csharp
public async Task<IActionResult> Index(long id = 1)
{
    return View(await context.Products.FindAsync(id));
}
```
x??

---

#### Creating a Razor View
Background context: To create a Razor view, you need to place the file in the appropriate directory and define it using HTML with embedded C# code.

:p How do you create a Razor view for an action method?
??x
Create a folder corresponding to the controller (e.g., `Views/Home/`) and add a `.cshtml` file with the same name as the action method. Use HTML tags and embed C# expressions to generate dynamic content.
```csharp
public async Task<IActionResult> Index(long id = 1)
{
    return View(await context.Products.FindAsync(id));
}
```
x??

---

#### Razor Syntax in Views
Background context: Razor syntax allows you to write C# code within your HTML views. This enables you to dynamically generate content based on the model passed to the view.

:p How do you use Razor syntax to insert dynamic data into a view?
??x
Use `@Model.Property` to insert the value of properties from the model.
```html
<tr>
    <th>Name</th>
    <td>@Model.Name</td>
</tr>
```
x??

---

#### Example of Dynamic Data in Views
Background context: The example provided demonstrates how dynamic data is inserted into a view using Razor syntax.

:p What does this line do: `<td>@Model.Price.ToString("c")</td>`?
??x
This line inserts the formatted price (as currency) from the model's `Price` property into the table cell.
```html
<tr>
    <th>Price</th>
    <td>@Model.Price.ToString("c")</td>
</tr>
```
x??

---

#### Adding New Elements to Views
Background context: Modifying views dynamically using Razor syntax allows you to add new elements or modify existing ones without changing the action method.

:p What is the impact of adding a new row in this view?
??x
Adding a new row with `@Model.CategoryId` inserts the category ID from the model into the table.
```html
<tr>
    <th>Category ID</th>
    <td>@Model.CategoryId</td>
</tr>
```
x??

---

#### Modifying a Razor View
Razor views allow you to generate dynamic content for your web application. When changes are made to the view, they need to be saved so that the application can detect them and reload the browser.

:p How do you modify and save a Razor view in ASP.NET Core?
??x
When modifying a Razor view, ensure that the changes are saved. The development server (`dotnet watch`) will automatically detect these changes and reload the browser to reflect the modifications. If no view file exists for a specified name or if the view cannot be found due to missing files, the application will throw an error.

To create a new view, add a `.cshtml` file in the appropriate folder (e.g., `Views/Home`). For instance, to create a view named `Watersports`, you would save it in the `Views/Home` directory with the content specified.
x??

---

#### Selecting a View by Name
Action methods can choose which Razor view to use by passing a name as an argument to the `View()` method. The view engine will look for this view in the default location (`Views/[controller]`) or, if not found, it will search in shared views (`Views/Shared`).

:p How does an action method select a specific Razor view using the `View()` method?
??x
An action method can select a specific Razor view by providing its name as an argument to the `View()` method. For example:

```csharp
public async Task<IActionResult> Index(long id = 1)
{
    Product? prod = await context.Products.FindAsync(id);
    if (prod?.CategoryId == 1) 
    {
        return View("Watersports", prod); // Selects the 'Watersports' view
    }
    else 
    {
        return View(prod); // Uses the default view for the model type
    }
}
```

Here, the `View()` method is called with a name and optionally a model. The view engine will then search for this view in the standard locations (`Views/Home` or `Views/Shared`). If it cannot find the view, an error will occur.
x??

---

#### Using Shared Views
Shared views allow you to create common content that can be reused across multiple controllers without duplicating code. They are stored in the `Views/Shared` folder.

:p How do shared views help reduce redundancy in your application?
??x
Shared views provide a way to store commonly used templates or sections of HTML that can be included in any Razor view, regardless of the controller it is associated with. By placing these shared components in the `Views/Shared` directory, you can avoid duplicating code and maintain consistency across your application.

For example, creating a common header or footer can be done once in a shared view file like `Common.cshtml`, which can then be included in other views as needed:

```csharp
public class HomeController : Controller
{
    // ...
    public IActionResult Common()
    {
        return View(); // Uses the 'Common' view from Views/Shared
    }
}
```

This reduces the risk of maintaining inconsistencies across different parts of your application.
x??

---

#### Specifying a View Location
If there is both a controller-specific and shared view with the same name, the Razor view engine will prioritize the controller-specific view. However, you can override this behavior by specifying the full path to the desired view file.

:p How do you specify the location of a shared view in an action method?
??x
You can explicitly specify the path to a shared view using the `View()` method with the complete path starting from the project root:

```csharp
public IActionResult Index()
{
    return View("/Views/Shared/Common.cshtml"); // Specifies full path to the 'Common' view
}
```

This ensures that even if there is a controller-specific view with the same name, the shared view will be used. However, using this method creates a hard dependency on a specific file location and should be done sparingly.
x??

---

