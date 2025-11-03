# Flashcards: Pro-ASPNET-Core-7_processed (Part 23)

**Starting Chapter:** 7.5 Styling the content. 7.5.1 Installing the Bootstrap package. 7.5.2 Applying Bootstrap styles

---

#### Installing Bootstrap Package
Background context: The process of adding and using the Bootstrap package to enhance the appearance of an application. The Bootstrap package is a popular front-end framework that provides pre-defined CSS styles, layout components, and JavaScript functions.

:p What are the steps to install the Bootstrap package in a .NET Core application using LibMan?

??x
To install the Bootstrap package in a .NET Core application using LibMan:
1. First, ensure you have installed the global `libman` tool by running:
   ```shell
   dotnet tool uninstall --global Microsoft.Web.LibraryManager.Cli
   dotnet tool install --global Microsoft.Web.LibraryManager.Cli --version 2.1.175
   ```
2. Then navigate to your application's root directory and run the following commands to initialize the example project and install Bootstrap:
   ```shell
   libman init -p cdnjs
   libman install bootstrap@5.2.3 -d wwwroot/lib/bootstrap
   ```

x??

---

#### Applying Bootstrap Styles in Layout File
Background context: The implementation of a classic two-column layout with a header using Bootstrap styles in the `_Layout.cshtml` file.

:p How does one apply Bootstrap CSS to the `_Layout.cshtml` file?

??x
To apply Bootstrap CSS, add the necessary HTML and link elements to include Bootstrap's stylesheet and define a common header. The code snippet is as follows:

```html
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>SportsStore</title>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
</head>
<body>
    <div class="bg-dark text-white p-2">
        <span class="navbar-brand ml-2">SPORTS STORE</span>
    </div>
    <div class="row m-1 p-1">
        <div id="categories" class="col-3">
            Put something useful here later
        </div>
        <div class="col-9">
            @RenderBody()
        </div>
    </div>
</body>
```

This snippet ensures that the Bootstrap styles are applied to all views that rely on this layout.

x??

---

#### Styling Content with Bootstrap
Background context: The process of styling content using Bootstrap classes in Razor views, particularly focusing on product listings and pagination links.

:p How does one style products listed in a Razor view?

??x
Styling products involves using Bootstrap's card, badge, and other utility classes. Here’s an example from the `Index.cshtml` file:

```csharp
@model ProductsListViewModel

@foreach (var p in Model.Products ?? Enumerable.Empty<Product>()) {
    <div class="card card-outline-primary m-1 p-1">
        <div class="bg-faded p-1">
            <h4>
                @p.Name
                <span class="badge rounded-pill bg-primary text-white" style="float:right">
                    <small>@p.Price.ToString("c")</small>
                </span>
            </h4>
        </div>
        <div class="card-text p-1">@p.Description</div>
    </div>
}
<div page-model="@Model.PagingInfo" page-action="Index"
     page-classes-enabled="true" page-class="btn"
     page-class-normal="btn-outline-dark" 
     page-class-selected="btn-primary" class="btn-group pull-right m-1"></div>
```

In this example, the `.card`, `.card-text`, and `.badge` classes are used to style individual product items. The badge is styled with a right float.

x??

---

#### Customizing Pagination Links
Background context: Implementing custom pagination link styles in Razor views using `PageLinkTagHelper`.

:p How does one customize the appearance of pagination links?

??x
To customize pagination links, define custom attributes on the div element that specify the Bootstrap classes. Here’s how to do it:

```csharp
using Microsoft.AspNetCore.Mvc;
using Microsoft.AspNetCore.Mvc.Rendering;
using Microsoft.AspNetCore.Mvc.Routing;
using Microsoft.AspNetCore.Mvc.ViewFeatures;
using Microsoft.AspNetCore.Razor.TagHelpers;

namespace SportsStore.Infrastructure
{
    [HtmlTargetElement("div", Attributes = "page-model")]
    public class PageLinkTagHelper : TagHelper
    {
        private IUrlHelperFactory urlHelperFactory;

        public PageLinkTagHelper(IUrlHelperFactory helperFactory)
        {
            urlHelperFactory = helperFactory;
        }

        [ViewContext]
        [HtmlAttributeNotBound]
        public ViewContext? ViewContext { get; set; }

        public PagingInfo? PageModel { get; set; }
        public string? PageAction { get; set; }
        public bool PageClassesEnabled { get; set; } = false;
        public string PageClass { get; set; } = String.Empty;
        public string PageClassNormal { get; set; } = String.Empty;
        public string PageClassSelected { get; set; } = String.Empty;

        public override void Process(TagHelperContext context, TagHelperOutput output)
        {
            if (ViewContext == null || PageModel == null)
            {
                IUrlHelper urlHelper = urlHelperFactory.GetUrlHelper(ViewContext);
                TagBuilder result = new TagBuilder("div");
                for (int i = 1; i <= PageModel.TotalPages; i++)
                {
                    TagBuilder tag = new TagBuilder("a");
                    tag.Attributes["href"] = urlHelper.Action(PageAction, new { productPage = i });
                    if (PageClassesEnabled)
                    {
                        // Add necessary Bootstrap classes here
                    }
                    result.InnerHtml.Append(tag);
                }
                output.Content.SetElementContent(result);
            }
        }
    }
}
```

This example shows how to use custom attributes and tag helper logic to generate styled pagination links. The `PageClass`, `PageClassNormal`, and `PageClassSelected` properties allow you to specify the Bootstrap classes.

x??

---

#### Tag Helper for Pagination
Tag helpers can dynamically generate HTML tags based on model data. The provided example uses tag helpers to create pagination links, applying different CSS classes depending on the current page status.

:p How does the tag helper determine which CSS class to apply to a pagination link?
??x
The tag helper determines the CSS class by checking if the current loop index `i` matches the `PageModel.CurrentPage`. If they match, it applies the `PageClassSelected` class; otherwise, it applies the `PageClassNormal` class.

Example:
```csharp
tag.AddCssClass(PageClassSelected);
tag.AddCssClass(i == PageModel.CurrentPage ? PageClassSelected : PageClassNormal);
```
x??

---
#### Styling with HTML Attributes and Tag Helpers
ASP.NET Core tag helpers allow for dynamic generation of HTML elements based on model data. The mapping between HTML attribute names (e.g., `page-class-normal`) and C# property names (`PageClassNormal`) is crucial.

:p How does ASP.NET Core handle the mapping between HTML attributes and C# properties in tag helpers?
??x
ASP.NET Core uses conventions to map HTML attribute names, such as `page-class-normal`, to corresponding C# property names, like `PageClassNormal`. This allows for more dynamic and flexible content generation without manually setting each attribute.

Example:
```csharp
tag.AddCssClass(PageClass);
tag.AddCssClass(i == PageModel.CurrentPage ? PageClassSelected : PageClassNormal);
```
x??

---
#### Partial View in ASP.NET Core
A partial view is a reusable piece of markup that can be embedded into another Razor view. This allows for reducing code duplication and maintaining consistency across multiple views.

:p What is the purpose of creating a partial view in an ASP.NET Core application?
??x
The purpose of creating a partial view is to encapsulate common or frequently used pieces of content, such as product summaries in this case, which can be reused throughout different parts of your application. This reduces code duplication and promotes cleaner, more maintainable code.

Example:
```csharp
@model ProductsListViewModel
@foreach (var p in Model.Products ?? Enumerable.Empty<Product>()) {
    <partial name="ProductSummary" model="p" />
}
```
x??

---
#### Using `partial` Tag Helper
The `partial` tag helper is used to include the content of another view file within the current view. It's particularly useful when you need similar markup in multiple places.

:p How do you use the `partial` tag helper in an ASP.NET Core Razor view?
??x
You can use the `partial` tag helper by specifying the name of the partial view and passing its model data using the `model` attribute. The following example demonstrates how to include a partial view named "ProductSummary" for each product in the list.

Example:
```csharp
@model ProductsListViewModel
@foreach (var p in Model.Products ?? Enumerable.Empty<Product>()) {
    <partial name="ProductSummary" model="p" />
}
```
x??

---
#### Updating Index.cshtml with Partial View
By refactoring `Index.cshtml` to use a partial view, you can maintain cleaner code and reuse the same markup across different views.

:p Why was the existing Razor markup moved to a separate partial view?
??x
The existing Razor markup was moved to a separate partial view named "ProductSummary" to reduce redundancy. This approach allows for maintaining a single source of truth for product summaries, making it easier to update and reuse this content throughout your application without duplicating code.

Example:
```csharp
@model ProductsListViewModel
@foreach (var p in Model.Products ?? Enumerable.Empty<Product>()) {
    <partial name="ProductSummary" model="p" />
}
```
x??

---
#### Application Appearance with Styling
Styling was applied to the SportsStore application using CSS classes, enhancing its visual appeal and user experience.

:p How did the application's appearance change after applying styles?
??x
The application's appearance improved by adding appropriate CSS classes to elements like pagination links. For instance, different styles were applied based on whether a link represented the current page or not, making the interface more intuitive and visually appealing.

Example:
```csharp
tag.AddCssClass(i == PageModel.CurrentPage ? PageClassSelected : PageClassNormal);
```
x??

---

---
#### ASP.NET Core Project Setup
Background context: The SportsStore project is created using the basic ASP.NET Core template, providing a foundation for building web applications. This template includes essential components and configurations to get started quickly.

:p What command or tool is used to create an ASP.NET Core project?
??x
The `dotnet new` command in the terminal is typically used to generate a new ASP.NET Core project with the basic template.
```csharp
dotnet new mvc -n SportsStore
```
x??
---

#### Entity Framework Core Integration
Background context: ASP.NET Core has close integration with Entity Framework Core, which simplifies data access and management in .NET applications. This allows for easy interaction with relational databases.

:p What is Entity Framework Core used for in ASP.NET Core projects?
??x
Entity Framework Core is primarily used for managing data in .NET applications by providing an object-oriented model of the database.
```csharp
using Microsoft.EntityFrameworkCore;
public class SportsStoreContext : DbContext {
    public DbSet<Product> Products { get; set; }
}
```
x??
---

#### Paginating Data
Background context: Paginating data allows users to navigate through large datasets, making it easier to manage and display information. This can be achieved by including the page number in both query string and URL path parameters.

:p How can pagination be implemented when querying a database using ASP.NET Core?
??x
Pagination can be implemented by including the `page` parameter in the request (either through the query string or URL path) and using this value to limit the number of records returned from the database.
```csharp
var products = context.Products.Skip((pageNumber - 1) * pageSize).Take(pageSize).ToList();
```
x??
---

#### Styling with CSS Frameworks
Background context: ASP.NET Core applications can leverage popular CSS frameworks like Bootstrap for styling. This allows developers to quickly apply professional and responsive designs without writing extensive CSS.

:p How can HTML content generated by an ASP.NET Core application be styled using a CSS framework such as Bootstrap?
??x
HTML content generated by an ASP.NET Core application can be styled using a CSS framework like Bootstrap by including the necessary Bootstrap CSS file in the project. This is typically done by referencing the Bootstrap CDN in the layout or views.
```html
<!DOCTYPE html>
<html>
<head>
    <link href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css" rel="stylesheet">
</head>
<body>
    <!-- Your HTML content here -->
</body>
</html>
```
x??
---

