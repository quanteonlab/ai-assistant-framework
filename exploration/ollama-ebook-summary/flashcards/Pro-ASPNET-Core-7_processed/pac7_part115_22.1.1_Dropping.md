# Flashcards: Pro-ASPNET-Core-7_processed (Part 115)

**Starting Chapter:** 22.1.1 Dropping the database

---

---
#### Razor Views Overview
Background context: Razor views are files that combine HTML and C# code expressions, allowing for dynamic content generation within web applications. The core idea is to leverage server-side logic within HTML templates.

:p What are Razor views?
??x
Razor views are files that integrate HTML markup with C# code. They enable the creation of dynamic web pages by combining static HTML with server-side logic and expressions.
x??

---
#### View Compilation Process
Background context: When a view is used, it gets compiled into a C# class. The methods in this class are responsible for generating the HTML content dynamically.

:p How does a Razor view get compiled?
??x
A Razor view is compiled into a C# class at runtime. This process converts the view's code and markup into a set of methods that generate the final HTML output.
x??

---
#### Selecting Views from Action Methods
Background context: In ASP.NET Core MVC, you can select a view to render based on the action method used in your controller. You can pass data through these methods as needed.

:p How do action methods choose which view to display?
??x
Action methods in controllers determine which Razor view to use by returning an `IActionResult`. This result can include a specific view name, or it can return a generic `View` method with the appropriate model.
```csharp
public IActionResult SomeAction()
{
    // Logic here
    return View("SpecificViewName", model);
}
```
x??

---
#### ViewModel and Type Safety
Background context: A ViewModel is often used in Razor views to provide a strongly-typed object that ensures type safety during development.

:p What is the role of a ViewModel in Razor views?
??x
A ViewModel acts as an intermediary between the controller and the view. It provides a strongly-typed object that enhances the maintainability and readability of the code by ensuring type safety.
x??

---
#### ViewBag for Passing Data
Background context: The `ViewBag` is a dynamic property that can be used to pass data from action methods to views without explicitly defining properties.

:p How does ViewBag work in passing data?
??x
The `ViewBag` is a dynamic object that allows you to pass data from the controller to the view. You can set properties on `ViewBag` directly, and these values will then be accessible within the Razor view.
```csharp
public IActionResult SomeAction()
{
    ViewBag.Message = "Hello, World!";
    return View();
}
```
x??

---
#### Temp Data for Passing Data
Background context: The `TempDataDictionary` is used to pass data between controllers or actions. It persists data across redirects.

:p What is TempData used for?
??x
The `TempDataDictionary` is used to store and retrieve data that needs to persist across a single redirect operation. This can be useful when you need to pass temporary data from one action method to another.
```csharp
public IActionResult SomeAction()
{
    TempData["Message"] = "This message persists for one redirect.";
    return RedirectToAction("OtherAction");
}
```
x??

---
#### Layouts and Sections in Views
Background context: Layouts provide a way to define common content that can be reused across multiple views. Sections allow you to inject custom content into these layouts.

:p What is the purpose of using layouts in Razor views?
??x
Layouts are used to define shared content that can be included in multiple views. This helps reduce redundancy and maintain consistency across your application.
```csharp
@{
    Layout = "_MyLayout";
}
```
x??

---
#### Creating Partial Views
Background context: Partial views are reusable view components that can be embedded within other views or layouts.

:p What is a partial view used for?
??x
Partial views are reusable snippets of HTML and C# code that can be included in multiple locations. They allow you to encapsulate common UI elements, making your application more modular.
```csharp
@Html.Partial("_MyPartialView", model)
```
x??

---
#### Encoding Data within Views
Background context: By default, Razor views encode data to prevent XSS attacks. However, there are scenarios where you might want to disable this encoding.

:p How does encoding work in Razor views?
??x
Razor views encode data by default to prevent Cross-Site Scripting (XSS) attacks. This means that any HTML tags within the data will be escaped as plain text.
```csharp
<p>@Html.Raw(Model.SomeData)</p>
```
x??

---
#### Disabling Encoding for Data in Views
Background context: While encoding is important for security, there are times when you might need to output raw data without escaping.

:p How can you disable encoding in Razor views?
??x
To disable encoding and output raw HTML or text, you can use the `Html.Raw` method. This allows you to display content that may contain dangerous characters.
```csharp
<p>@Html.Raw(Model.SomeDangerousData)</p>
```
x??

---
#### JSON Encoding within Views
Background context: Razor views also handle the encoding of JSON data. By default, JSON objects are properly formatted and safe for output.

:p How does Razor handle JSON in views?
??x
Razor views automatically encode JSON data to ensure it is safely rendered on the client side without causing any issues.
```csharp
<p>@JsonHelper.Encode(Model.SomeJsonData)</p>
```
x??

---

---
#### Enabling Sessions in Program.cs
Background context: The chapter introduces enabling sessions for managing user data across requests. This is achieved by using services like `AddSession` and configuring them properly.

:p How do you enable session management in a .NET Core application?
??x
To enable session management, the following code should be added to the `Program.cs` file:

```csharp
builder.Services.AddDistributedMemoryCache(); // Adds an in-memory distributed cache.
builder.Services.AddSession(options => {
    options.Cookie.IsEssential = true; // Makes sure sessions are considered essential for user interaction.
});

// Then, use session middleware before controllers:
app.UseSession();
```

x??
---
#### Dropping the Database Using `dotnet ef`
Background context: To ensure a clean slate when running examples, you might need to drop the existing database. The `dotnet ef` command provides a way to do this.

:p How can you drop an existing database using the `dotnet ef` command?
??x
To drop the existing database, run the following command in a PowerShell prompt:

```sh
dotnet ef database drop --force
```

The `--force` option is used to bypass any prompts and force the operation.

x??
---
#### Running the Example Application with `dotnet watch`
Background context: The `dotnet watch` command allows developers to run an application while also enabling hot reloads for views. However, if a change cannot be handled by a simple reload, it will prompt to restart the app.

:p How do you start the example application using `dotnet watch`?
??x
To start the example application with `dotnet watch`, use the following command in a PowerShell prompt:

```sh
dotnet watch
```

This command monitors changes and applies hot reloads where possible. If a change cannot be handled by a simple reload, it will display a message asking whether to restart the app.

x??
---
#### Using ViewBag for Dynamic Data Passing
Background context: `ViewBag` is a dynamic object that can hold additional data beyond what is passed through the view model. It's useful when you need to pass more dynamic or calculated information to your views.

:p How does an action method use `ViewBag` to pass extra data to a view?
??x
An action method uses `ViewBag` to assign values to dynamic properties, which are then accessible in the corresponding view via ViewBag. Here's how it works:

```csharp
public async Task<IActionResult> Index(long id = 1)
{
    ViewBag.AveragePrice = await context.Products.AverageAsync(p => p.Price);
    return View(await context.Products.FindAsync(id));
}
```

In this example, `ViewBag` is used to store the average product price. This value can then be accessed in the view via `@ViewBag.AveragePrice`.

x??
---

#### Using ViewBag for Providing Supplementary Data

Background context: The `ViewBag` is a dynamic object that can be used to pass data from controllers to views. It provides an easy way to add small amounts of supplementary data without creating new view model classes.

The main advantage of `ViewBag` is its simplicity and flexibility, but it lacks type safety because the compiler cannot check the properties on dynamic objects. This means you might face runtime errors if you mistakenly use a non-existent property or pass incorrect data types.

:p When should we consider using ViewBag over creating new view models?
??x
The decision to use `ViewBag` primarily depends on whether the data required by the view is minimal and does not require strong typing. If the same view bag property is used across multiple actions, or if an action adds more than two or three properties to it, consider defining a custom view model instead.

For example:
```csharp
public class CubeViewModel {
    public string Value { get; set; }
    public string Result { get; set; }
}
```
Using this model in the controller and view ensures better code organization and type safety.
x??

---

#### Understanding TempData for Persistent Data Across Requests

Background context: `TempData` is a feature that allows you to store data temporarily between requests. It's particularly useful when performing redirects, as it persists through the redirect and can be accessed by subsequent actions.

The `TempData` dictionary stores values using string keys and supports only string-serializable objects. Once read, these values are marked for deletion but removed at the end of the request processing cycle.

:p How does TempData differ from ViewBag in terms of data persistence?
??x
`TempData` provides a way to store and pass data between actions, especially during redirects. Unlike `ViewBag`, which is dynamically typed and not type-safe, `TempData` stores values as key-value pairs in a dictionary and marks them for removal upon reading.

Here's an example:
```csharp
public IActionResult Cube(double num)
{
    TempData["value"] = num.ToString();
    TempData["result"] = Math.Pow(num, 3).ToString();
    return RedirectToAction(nameof(Index));
}
```
In this case, the `TempData` values are preserved across the redirect to the `Index` action.

Once the data is accessed in the view:
```csharp
@if (TempData["result"] != null)
{
    <div class="bg-info text-white m-2 p-2">
        The cube of @TempData["value"] is @TempData["result"]
    </div>
}
```
This ensures that the data remains available until it's read, after which it will be removed.
x??

---

#### Implementing TempData with a Controller Example

Background context: In this example, we create a `CubedController` to demonstrate how to use `TempData`. The controller includes an `Index` method for rendering a view and a `Cube` action that performs calculations and stores results using `TempData`.

:p How does the `CubedController` store data temporarily before redirecting?
??x
The `Cube` action in `CubedController` uses `TempData` to store both input (`num`) and result values. This ensures that these values are available after a redirection.

Here’s how it works:
```csharp
public IActionResult Cube(double num)
{
    TempData["value"] = num.ToString();  // Store the input value
    TempData["result"] = Math.Pow(num, 3).ToString();  // Store the result of the calculation
    return RedirectToAction(nameof(Index));  // Redirect to the Index action
}
```
Upon redirection, `TempData` persists these values and makes them accessible in the target view.

In the view (`Cubed.cshtml`), you can access these stored values:
```csharp
<div class="bg-info text-white m-2 p-2">
    The cube of @TempData["value"] is @TempData["result"]
</div>
```
These values are available until they are read, after which they are marked for deletion.
x??

---

#### Managing TempData Values with Peek and Keep

Background context: `TempData` allows you to manage data persistence through methods like `Peek` (to inspect without removing) and `Keep` (to prevent removal).

:p What does the `TempData` property provide that can help in managing data?
??x
The `TempData` property offers additional methods such as `Peek` and `Keep`. The `Peek` method allows you to check if a value exists without removing it, while the `Keep` method prevents a previously read value from being removed.

Here’s an example:
```csharp
if (TempData.Peek("value") != null)
{
    // Perform operations using the value
}
```
To keep a value in `TempData`, you can use:
```csharp
TempData.Keep("value");
```

This approach ensures that the data remains available for multiple requests while maintaining control over its removal.
x??

---

