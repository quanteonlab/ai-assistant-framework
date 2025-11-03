# Flashcards: Pro-ASPNET-Core-7_processed (Part 73)

**Starting Chapter:** 8.1.4 Correcting the page count

---

#### Highlighting Selected Categories
Background context: The provided `Default.cshtml` file is part of a navigation menu that dynamically highlights selected categories. It uses Razor expressions to conditionally apply Bootstrap button classes based on the current category.

:p What does the Razor expression within the class attribute do in the `Default.cshtml` file?
??x
The Razor expression checks if the current category matches the `ViewBag.SelectedCategory`. If they match, it applies the "btn-primary" class; otherwise, it uses "btn-outline-secondary". This changes the appearance of the button to indicate which category is currently selected.

```html
<a class="@(category == ViewBag.SelectedCategory ? "btn-primary": "btn-outline-secondary")"
```
x??

---
#### Correcting Page Count for Category-Specific Navigation
Background context: The `Index` action method in the `HomeController.cs` file determines how many pages of products to display. Previously, it calculated based on all products, leading to incorrect page navigation when a specific category is selected.

:p How does the updated `Index` action method handle pagination differently for categories?
??x
The updated `Index` action method now takes into account the selected category by checking if a category is provided (`category != null`). If it is, the method calculates the total items based on that category; otherwise, it counts all products. This ensures correct page links when navigating within a specific category.

```csharp
public ViewResult Index(string? category, int productPage = 1)
{
    return View(new ProductsListViewModel 
    {
        Products = repository.Products
            .Where(p => category == null || p.Category == category)
            .OrderBy(p => p.ProductID)
            .Skip((productPage - 1) * PageSize)
            .Take(PageSize),
        PagingInfo = new PagingInfo 
        { 
            CurrentPage = productPage, 
            ItemsPerPage = PageSize,
            TotalItems = category == null
                ? repository.Products.Count()
                : repository.Products.Where(e => e.Category == category).Count() 
        },
        CurrentCategory = category
    });
}
```
x??

---
#### Unit Testing Category-Specific Product Counts
Background context: To ensure the `Index` action method accurately returns product counts for different categories, a unit test was created. It uses a mock repository with known data to verify the correct count is returned.

:p How does the unit test method validate category-specific product counts?
??x
The unit test method creates a mock repository with predefined products and then calls the `Index` action method with each category in turn. By asserting the correct count of products for each category, it ensures the pagination logic works as expected.

```csharp
[Fact]
public void Generate_Category_Specific_Product_Count()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    Product[] products = 
        { 
            new Product {ProductID = 1, Name = "P1", Category = "Cat1"}, 
            new Product {ProductID = 2, Name = "P2", Category = "Cat2"}, 
            new Product {ProductID = 3, Name = "P3", Category = "Cat1"}, 
            new Product {ProductID = 4, Name = "P4", Category = "Cat2"}, 
            new Product {ProductID = 5, Name = "P5", Category = "Cat3"} 
        };
    mock.Setup(m => m.Products).Returns(products.AsQueryable());
    HomeController target = new HomeController(mock.Object);
    target.PageSize = 3;

    // Act & Assert
    var result = target.Index("Cat1") as ViewResult;
    ProductsListViewModel model = result.Model as ProductsListViewModel;
    Assert.Equal(2, model.PagingInfo.TotalItems);

    result = target.Index("Cat2") as ViewResult;
    model = result.Model as ProductsListViewModel;
    Assert.Equal(2, model.PagingInfo.TotalItems);

    result = target.Index("Cat3") as ViewResult;
    model = result.Model as ProductsListViewModel;
    Assert.Equal(1, model.PagingInfo.TotalItems);
}
```
x??

---
---

---
#### Navigation and Cart Result Handling
Background context: The provided code snippet is part of a test for handling navigation and cart functionalities within a web application. It involves navigating to different categories and checking the total items count, ensuring consistency across different category pages.

:p What does `result?.ViewData?.Model as ProductsListViewModel;` do in this context?
??x
This line retrieves the model data from the result of a navigation action, specifically casting it to a `ProductsListViewModel`. This is used to access properties like `PagingInfo`, which contains information about the current page's items and total count.

```csharp
// Pseudo-code for understanding the context
if (result != null && result.ViewData != null) {
    var viewModel = result.ViewData.Model as ProductsListViewModel;
    if (viewModel != null) {
        int? totalCount = viewModel.PagingInfo.TotalItems;
        // Use totalCount in assertions or further logic
    }
}
```
x??

---
#### Index Method with Different Parameters
Background context: The `Index` method is called multiple times with different parameters to retrieve and verify the total number of items in different categories. This helps ensure that the correct counts are displayed for each category.

:p How does calling `GetModel(target.Index("Cat1"))?.PagingInfo.TotalItems;` work?
??x
This line retrieves the model data from a specific category index action (`"Cat1"`), casts it to a `ProductsListViewModel`, and then accesses the `TotalItems` property in its `PagingInfo`. This ensures that the total number of items for that particular category is correctly retrieved.

```csharp
// Pseudo-code for understanding the context
int? res = GetModel(target.Index("Cat1"))?.PagingInfo.TotalItems;
if (res.HasValue) {
    // Use res in assertions or further logic
}
```
x??

---
#### Adding Razor Pages to SportsStore
Background context: The provided code snippet configures the `Program.cs` file for enabling Razor Pages in the SportsStore application. This is done by adding necessary services and configuring routing.

:p What does `builder.Services.AddRazorPages();` do in this configuration?
??x
This line adds the necessary services to support Razor Pages within the application. It sets up the environment so that Razor Page controllers, views, and other related components can be used effectively.

```csharp
// Pseudo-code for understanding the context
builder.Services.AddRazorPages();
```
x??

---
#### Configuring Static Files
Background context: The `app.UseStaticFiles();` line in the `Program.cs` file configures the application to serve static files, which are typically used for serving assets like images, CSS, and JavaScript.

:p What is the purpose of `app.UseStaticFiles();`?
??x
The purpose of `app.UseStaticFiles();` is to enable the web application to serve static files from a specified directory. This allows for easy access to resources such as images, stylesheets, and scripts that are stored in a static folder.

```csharp
// Pseudo-code for understanding the context
app.UseStaticFiles();
```
x??

---
#### Mapping Routing for Razor Pages
Background context: The provided code snippet includes routing configurations for different paths, including support for Razor Pages. This ensures that requests to specific URLs are correctly routed to the appropriate controllers or pages.

:p How does `app.MapRazorPages();` contribute to routing in this application?
??x
The line `app.MapRazorPages();` registers Razor Pages as endpoints that the URL routing system can use to handle requests. This allows for dynamic content generation based on user interactions and routes defined within Razor Page files.

```csharp
// Pseudo-code for understanding the context
app.MapRazorPages();
```
x??

---
#### Ensuring Data Populated in SportsStore
Background context: The `SeedData.EnsurePopulated(app);` line ensures that sample data is populated into the database, which is necessary for testing and development purposes. This helps in verifying the application's functionality with actual data.

:p What does `SeedData.EnsurePopulated(app);` do?
??x
This method seeds the database with initial or sample data if it has not already been populated. It ensures that there are records available to test the application, providing a starting point for development and testing phases.

```csharp
// Pseudo-code for understanding the context
SeedData.EnsurePopulated(app);
```
x??

---

---
#### _ViewImports.cshtml File Setup
Background context: The `_ViewImports.cshtml` file is used to set global configurations for Razor Pages within a specific folder. It allows you to define namespaces, add tag helpers, and specify layout files for all views in that folder.

:p What does the `_ViewImports.cshtml` file do?
??x
The `_ViewImports.cshtml` file sets up common configurations like namespaces, tag helper usages, and default layouts for Razor Pages within a specific folder. For example:
```cshtml
@namespace SportsStore.Pages
@using Microsoft.AspNetCore.Mvc.RazorPages
@using SportsStore.Models
@using SportsStore.Infrastructure
@addTagHelper *, Microsoft.AspNetCore.Mvc.TagHelpers
```
It also sets the layout file that all pages in this folder will use by default, such as `_CartLayout`:
```cshtml
@{ Layout = "_CartLayout"; }
```
x??

---
#### _ViewStart.cshtml File for Default Layout
Background context: The `_ViewStart.cshtml` file is a special configuration file used by Razor Pages to set the default layout for all pages within a specific folder. This allows you to specify shared elements like headers, footers, and navigation that should be applied across multiple views.

:p What does the `_ViewStart.cshtml` file configure?
??x
The `_ViewStart.cshtml` file configures the default layout that will be used by all Razor Pages in a specific folder. In this case, it sets the default layout to `_CartLayout`:
```cshtml
@{
    Layout = "_CartLayout";
}
```
This means every page within the `Pages` folder will use the `_CartLayout.cshtml` as its base template.

x??

---
#### _CartLayout.cshtml File Content
Background context: The `_CartLayout.cshtml` file is a layout file used by Razor Pages to define common structure and elements that should be shared across multiple views. It includes necessary tags, meta information, and placeholders for dynamic content.

:p What does the `_CartLayout.cshtml` file contain?
??x
The `_CartLayout.cshtml` file contains HTML structure with some essential metadata and a placeholder for rendering dynamic content:
```cshtml
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
    <div class="m-1 p-1">
        @RenderBody()
    </div>
</body>
</html>
```
It sets up a basic HTML structure with Bootstrap styles and includes the `SPORTS STORE` brand in the header. The `@RenderBody()` directive is where the actual content of each Razor Page will be rendered.

x??

---
#### Cart.cshtml File for Shopping Cart
Background context: The `Cart.cshtml` file is a Razor Page that defines a specific page within your application, allowing you to handle shopping cart-related operations like adding items and displaying them. It uses a `.cshtml` file for the view and a corresponding `.cshtml.cs` class for handling logic.

:p What does the `Cart.cshtml` file do?
??x
The `Cart.cshtml` file is a Razor Page that displays content related to a shopping cart. By default, it shows placeholder text indicating its functionality:
```cshtml
@page
<h4>This is the Cart Page</h4>
```
This page uses the layout specified in `_ViewStart.cshtml` and can be accessed via URL `http://localhost:5000/cart`.

x??

---
#### UrlExtensions.cs File for Generating URLs
Background context: The `UrlExtensions.cs` file contains custom extension methods to generate URLs that include query strings. This is useful when you need to redirect users back to the same page after performing an operation, such as adding a product to the cart.

:p What does the `PathAndQuery` method do?
??x
The `PathAndQuery` method generates a URL by combining the path and query string from an HTTP request. This is useful for generating URLs that include parameters:
```cshtml
public static class UrlExtensions {
    public static string PathAndQuery(this HttpRequest request) =>
        request.QueryString.HasValue
            ? $"{request.Path}{request.QueryString}"
            : request.Path.ToString();
}
```
For example, if the current path is `/products` and there is a query string `?page=2`, it will return `/products?page=2`.

x??

---
#### Product Summary Partial View for Add to Cart Buttons
Background context: The partial view `ProductSummary.cshtml` is used to display product details on the website. It needs to include an "Add To Cart" button that can submit a form to the cart page when clicked.

:p How does the `ProductSummary.cshtml` file handle adding products to the cart?
??x
The `ProductSummary.cshtml` partial view includes a form with hidden inputs and a button to add products to the cart. It uses ASP.NET Core tag helpers for generating forms:
```cshtml
@model Product

<div class="card card-outline-primary m-1 p-1">
    <div class="bg-faded p-1">
        <h4>
            @Model.Name
            <span class="badge rounded-pill bg-primary text-white" style="float:right">
                <small>@Model.Price.ToString("c")</small>
            </span>
        </h4>
    </div>
    <form id="@Model.ProductID" asp-page="/Cart" method="post">
        <input type="hidden" asp-for="ProductID" />
        <input type="hidden" name="returnUrl" value="@ViewContext.HttpContext.Request.PathAndQuery()" />
        <span class="card-text p-1">
            @Model.Description
            <button type="submit" style="float:right" class="btn btn-success btn-sm pull-right">
                Add To Cart
            </button>
        </span>
    </form>
</div>
```
The form submits to the `/Cart` page and includes hidden fields for the product ID and return URL, ensuring that users are redirected back to their original location after adding an item.

x??

---

