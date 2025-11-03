# Flashcards: Pro-ASPNET-Core-7_processed (Part 116)

**Starting Chapter:** 22.4 Working with layouts

---

#### TempData and ViewData Usage
Background context explaining the use of `TempData` and `ViewData`. These properties are used to store data between requests. `TempData` stores values temporarily across a single request and subsequent request, making it suitable for scenarios like showing messages or redirecting with data.

:p What is the difference between `TempData` and `ViewData` in terms of usage?
??x
The main difference lies in their lifetime. `TempData` keeps the data available until read once (across a single request and its subsequent request), while `ViewData` persists only for the current request.

For example, if you want to display a message after a user logs out, `TempData` would be appropriate because it ensures that the message is shown on the next page load but not persist any longer than necessary. On the other hand, `ViewData` could be used for data that changes frequently within the same request.

```csharp
public class HomeController : Controller {
    public IActionResult LogOut() {
        // Store a logout message in TempData
        TempData["LogoutMessage"] = "You have been logged out successfully.";

        // Redirect to another action method or view
        return RedirectToAction("Index");
    }

    public IActionResult Index() {
        var message = TempData["LogoutMessage"];
        // Use the message in your view as needed
        return View();
    }
}
```
x??

---

#### Razor Layouts and Shared Content
Background context on how layouts can consolidate common HTML content across multiple views. Using a layout simplifies maintenance by centralizing repeated elements like header, footer, or CSS imports.

:p What is the purpose of using a layout in ASP.NET Core MVC?
??x
The primary purpose of using a layout is to avoid duplicating HTML code across multiple views. This helps maintain consistency and ease of updates. For example, if you have a common navigation bar, logo, or footer that needs to be present on every page, placing this content in a shared layout file ensures it's automatically included wherever the layout is referenced.

You can create a layout by creating an `_Layout.cshtml` file in the `Views/Shared` folder and defining common sections like headers, footers, etc. Then, you can apply this layout to any view that needs it by setting the `Layout` property in the view's Razor code block.

```csharp
@{
    Layout = "_Layout";
}
```

This snippet sets the current view's layout to `_Layout.cshtml`. The layout file is searched in both the controller-specific and shared views folders.

Example of a simple layout:

```cshtml
<!-- Views/Shared/_Layout.cshtml -->
<!DOCTYPE html>
<html>
<head>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
    <h6 class="bg-primary text-white text-center m-2 p-2">Shared View</h6>
    @RenderBody()
</body>
</html>
```

Using this layout, views can focus on their unique content without worrying about common elements.

```cshtml
<!-- Views/Home/Index.cshtml -->
@model Product?
@{
    Layout = "_Layout";
}

<div class="m-2">
    <table class="table table-sm table-striped table-bordered">
        <tbody>
            <tr><th>Name</th><td>@Model?.Name</td></tr>
            <tr>
                <th>Price</th>
                <td>
                    @Model?.Price.ToString("c")
                    (@(((Model?.Price / ViewBag.AveragePrice) * 100).ToString("F2")) percent of average price)
                </td>
            </tr>
            <tr><th>Category ID</th><td>@Model?.CategoryId</td></tr>
        </tbody>
    </table>
</div>
```
x??

---

#### Configuring Layouts Using ViewBag
Background context: When working with layouts, a view can provide specific data values to customize common content. The `ViewBag` allows you to pass these values from the view to the layout file.

:p How does a view use `ViewBag` to configure a layout?
??x
A view uses `ViewBag` properties to customize the layout. For example, in an `Index.cshtml` file within the `Views/Home` folder:
```csharp
@model Product?

{ 
    Layout = "_Layout"; 
    ViewBag.Title = "Product Table"; 
}
```
The above code snippet sets the `Title` property which can be used in the layout. This allows for dynamic content customization without hard-coding values directly into the layout file.

This approach uses:
- `ViewBag.Title`: A dynamic title that can vary based on the view context.
x??

---

#### Using ViewBag Property in Layout
Background context: The `ViewBag` property can be used both in views and layouts. Views pass data to layouts, but layouts cannot rely solely on these properties being defined.

:p How does a layout use `ViewBag` for dynamic content?
??x
A layout uses `ViewBag` properties to dynamically generate content based on the view's context. For instance, in the `_Layout.cshtml` file:
```csharp
<title>@ViewBag.Title</title>
```
This code snippet will insert the value of `ViewBag.Title` into the HTML title element. If no `Title` property is set by a view, it defaults to "Layout".

To ensure fallback values are provided when necessary, use conditional expressions like:
```csharp
@(ViewBag.Title ?? "Default Title")
```
This ensures that if `ViewBag.Title` is not defined in the view, "Default Title" will be used instead.

If no title is set by either the view or layout:
- The expression evaluates to `"Default Title"` as a fallback.
x??

---

#### Setting Up a Default Layout with View Start File
Background context: To avoid setting the `Layout` property in every view file, you can create a default layout using a view start file.

:p How does creating a `_ViewStart.cshtml` file help manage layouts?
??x
Creating a `_ViewStart.cshtml` file at the root of your views folder allows you to set a default layout for all your views. For example:
```csharp
@{
    Layout = "_Layout";
}
```
This line sets the `Layout` property for all view files, ensuring they use the same layout unless explicitly overridden.

By adding this file and setting it up as described, views can focus on their specific content without needing to specify a layout. This simplifies managing common layout elements across multiple pages.
x??

---

#### Removing Common Content from Shared Views
Background context: Once you have set a default layout with `_ViewStart.cshtml`, shared views no longer need to define repetitive common content.

:p How does removing common view content work in practice?
??x
Once the default layout is defined, shared views can remove redundant code. For example, in `Common.cshtml`:
```csharp
<h6 class="bg-secondary text-white text-center m-2 p-2">Shared View</h6>
```
This view no longer needs to set the `Layout` property since it uses the default layout defined by `_ViewStart.cshtml`.

The result is that the content in `Common.cshtml` will be added directly into the body section of the HTML response, inheriting the structure from the default layout.
x??

---

#### Overriding Default Layouts
Background context: In ASP.NET Core MVC, views can override the layout specified by a view start file. This is useful when different views require distinct layouts or no layout at all.

:p How can a view select a specific layout?
??x
A view can specify its own layout by assigning a string to the `Layout` property in the Razor code block.
```razor
@{
    Layout = "_ImportantLayout";
}
```
This value overrides any default layout set by a view start file. For instance, if `_ImportantLayout.cshtml` is used as shown in Listing 22.15, it will be applied to views that explicitly override the default layout.
x??

---
#### Selecting Layouts Programmatically
Background context: The `Layout` property can also be set based on expressions within a view. This allows for dynamic selection of layouts depending on the data model.

:p How can a layout be selected conditionally in a Razor view?
??x
A conditional expression can be used to select different layouts based on properties of the model or other conditions.
```razor
@{
    Layout = Model.Price > 100 ? "_ImportantLayout" : "_Layout";
}
```
In this example, if `Model.Price` is greater than 100, `_ImportantLayout` will be used; otherwise, `_Layout` will be applied. This flexibility allows for customizing the layout based on the specific view content.
x??

---
#### Disabling Layouts in Complete HTML Documents
Background context: When a view contains an entire HTML document (e.g., for pages that don’t need any additional structure), setting `Layout = null;` disables further layouts, preventing duplicate structural elements.

:p How can you disable layout in a view that already has an HTML document?
??x
To prevent the Razor engine from applying any more layout to a view that contains its own complete HTML structure, set the `Layout` property to `null`.
```razor
@{
    Layout = null;
}
```
This is particularly useful when the view itself handles all aspects of rendering and does not need additional wrapping in another layout. It ensures that only the content defined by this view is rendered.
x??

---

