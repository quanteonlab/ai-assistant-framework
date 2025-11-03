# Flashcards: Pro-ASPNET-Core-7_processed (Part 25)

**Starting Chapter:** 8.1.4 Correcting the page count

---

#### Highlighting Selected Categories
Background context: The provided code snippet demonstrates how to highlight selected categories using Bootstrap classes. This is achieved by dynamically applying the `btn-primary` class to the currently selected category and `btn-outline-secondary` to other categories.
:p How does the code highlight the selected category?
??x
The code uses Razor syntax within the `class` attribute of an anchor tag to conditionally apply Bootstrap styling based on whether the current category matches the selected category. If it matches, the `btn-primary` class is applied; otherwise, `btn-outline-secondary`.
```html
@foreach (string category in Model ?? Enumerable.Empty<string>()) {
    <a class="btn @(category == ViewBag.SelectedCategory ? "btn-primary" : "btn-outline-secondary")"
       asp-action="Index" asp-controller="Home" 
       asp-route-category="@category"
       asp-route-productPage="1">
        @category
    </a>
}
```
x??

---

#### Correcting Page Count for Selected Categories
Background context: The provided code snippet aims to adjust the pagination logic so that it correctly reflects the number of products in a selected category. This ensures users do not end up on an empty page if they navigate through multiple pages.
:p How does the Index action method update to handle categories properly?
??x
The `Index` action method now takes into account the selected category when determining pagination information. If no category is specified, it uses the total number of products; otherwise, it calculates based on the current category’s product count.
```csharp
public ViewResult Index(string? category, int productPage = 1)
{
    return View(new ProductsListViewModel {
        Products = repository.Products
            .Where(p => category == null || p.Category == category)
            .OrderBy(p => p.ProductID)
            .Skip((productPage - 1) * PageSize)
            .Take(PageSize),
        PagingInfo = new PagingInfo {
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

#### Unit Test for Category-Specific Product Counts
Background context: The provided unit test ensures that the application can correctly count products in different categories. This is crucial for accurate pagination and display of product lists.
:p How does the unit test verify category-specific product counts?
??x
The unit test creates a mock repository with predefined data across multiple categories. It then calls the `Index` action method for each category, verifying that the returned product count matches the expected value.
```csharp
[Fact]
public void Generate_Category_Specific_Product_Count()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    Product[] products = {
        new Product {ProductID = 1, Name = "P1", Category = "Cat1"},
        new Product {ProductID = 2, Name = "P2", Category = "Cat2"},
        new Product {ProductID = 3, Name = "P3", Category = "Cat1"},
        new Product {ProductID = 4, Name = "P4", Category = "Cat2"},
        new Product {ProductID = 5, Name = "P5", Category = "Cat3"}
    };
    mock.Setup(m => m.Products).Returns(products.AsQueryable<Product>());
    HomeController target = new HomeController(mock.Object);
    target.PageSize = 3;

    // Act
    var result1 = target.Index("Cat1");
    var model1 = ((ViewResult)result1).Model as ProductsListViewModel;
    int countCat1 = model1.PagingInfo.TotalItems;

    var result2 = target.Index("Cat2");
    var model2 = ((ViewResult)result2).Model as ProductsListViewModel;
    int countCat2 = model2.PagingInfo.TotalItems;

    // Assert
    Assert.Equal(3, countCat1);  // Cat1 has 3 products
    Assert.Equal(2, countCat2);  // Cat2 has 2 products
}
```
x??

---

---
#### Navigation and Cart Check
Background context: This section covers checking navigation and cart functionality within a sports store application. The code snippet provided demonstrates retrieving and verifying data for different categories and an overall count of products.

:p What is the purpose of the `GetModel` method in this context?
??x
The `GetModel` method serves to retrieve the model associated with a specific category or page index. It checks if the `ViewData.Model` is not null, then casts it to a `ProductsListViewModel`. This ensures that the correct data for the specified category (or all categories) is being fetched and can be used for assertions.

```csharp
int? res1 = GetModel(target.Index("Cat1"))?.PagingInfo.TotalItems;
```
x??

---
#### ShoppingCart Flow Explanation
Background context: The text describes the basic flow of a shopping cart in an online store. It mentions that adding items to the cart, viewing a summary, and checking out are key functionalities.

:p What does the Add To Cart button do?
??x
The Add To Cart button is responsible for allowing users to add products from the catalog to their cart. When clicked, it typically updates the shopping cart state and may display a summary of the items in the cart along with the total cost.

x??

---
#### Configuring Razor Pages
Background context: This section explains how to configure Razor Pages within an ASP.NET Core application for implementing a shopping cart feature. The `Program.cs` file is used to set up services required by Razor Pages and map them appropriately.

:p How does the `AddRazorPages` method in Program.cs enable Razor Pages?
??x
The `AddRazorPages` method registers the necessary services for Razor Pages, including configuration settings and dependencies needed to run pages. This enables the application to handle requests using Razor Pages, which can be accessed via URLs.

```csharp
builder.Services.AddRazorPages();
```
x??

---
#### Enabling Razor Pages in SportsStore Application
Background context: The text describes setting up the `Program.cs` file for enabling Razor Pages and configuring them properly. This includes adding necessary services and routes to handle requests.

:p What does the `MapRazorPages` method do?
??x
The `MapRazorPages` method registers Razor Pages as endpoints that can be handled by the URL routing system. This allows requests matching specific patterns (like `/cart` or `/summary`) to be routed to corresponding Razor Page handlers.

```csharp
app.MapRazorPages();
```
x??

---
#### _ViewImports.cshtml File Setup
Background context: The text mentions adding an `_ViewImports.cshtml` file to the `Pages` folder, setting up namespaces and allowing for easier use of application classes within views.

:p What is the purpose of the `_ViewImports.cshtml` file?
??x
The `_ViewImports.cshtml` file sets default namespaces used across all Razor Pages in a project. This simplifies referencing classes without needing to specify their full namespace, enhancing code readability and maintainability.

```csharp
@using SportsStore.Models
```
x??

---

#### Razor Pages Configuration and Layout Setup
Background context: In this section, we explore how to set up Razor Pages for a web application called SportsStore. This involves creating necessary files like `_ViewImports.cshtml`, `_ViewStart.cshtml`, and `_CartLayout.cshtml`. These files help in defining namespaces, adding tag helpers, and setting the default layout for Razor Pages.

:p What is the purpose of the _ViewImports.cshtml file?
??x
The `_ViewImports.cshtml` file is used to add global namespaces that are needed across all Razor views. This includes adding necessary namespaces like `Microsoft.AspNetCore.Mvc.RazorPages`, which contains the Tag Helpers and other utilities required for building the web application.

```csharp
@namespace SportsStore.Pages
@using Microsoft.AspNetCore.Mvc.RazorPages
@using SportsStore.Models
@using SportsStore.Infrastructure
@addTagHelper *, Microsoft.AspNetCore.Mvc.TagHelpers
```

x??

---

#### Razor Page Creation Using Visual Studio
Background context: This topic explains how to create a Razor Page in Visual Studio by utilizing the built-in template. The goal is to generate a basic structure for handling shopping cart operations.

:p How do you create a Razor Page named `Cart` using Visual Studio?
??x
To create a Razor Page named `Cart` using Visual Studio, follow these steps:
1. Open your project in Visual Studio.
2. Right-click on the `Pages` folder.
3. Select "Add" -> "New Scaffolded Item..." from the context menu.
4. Choose "Razor Page" as the item type.
5. Set the file name to `Cart.cshtml`.
6. Click "Add".

This process will generate both a `Cart.cshtml` view and a corresponding code-behind class named `Cart.cshtml.cs`.

x??

---

#### Adding Cart Layout
Background context: A layout file is essential for maintaining consistency across multiple Razor Pages. The `_CartLayout.cshtml` file defines the overall structure of pages using this layout.

:p What does the _CartLayout.cshtml file define?
??x
The `_CartLayout.cshtml` file defines a layout template used by Razor Pages in the SportsStore project. This layout includes common elements like the header, navigation, and footer, ensuring that all pages share a consistent look and feel.

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
    <div class="m-1 p-1">
        @RenderBody()
    </div>
</body>
</html>
```

x??

---

#### Implementing Add to Cart Buttons
Background context: To enable users to add products to the cart, we need to create buttons in the product summary. This involves modifying a partial view and adding JavaScript logic.

:p How do you modify the ProductSummary.cshtml file to include "Add to Cart" buttons?
??x
To include "Add to Cart" buttons in the `ProductSummary.cshtml` file, follow these steps:
1. Add a form element that submits the product ID.
2. Include hidden input elements for the product ID and return URL.
3. Use tag helpers to generate the form.

```csharp
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
            <button type="submit" style="float:right" class="btn btn-success btn-sm pull-right">Add To Cart</button>
        </span>
    </form>
</div>
```

x??

---

#### PathAndQuery Extension Method
Background context: The `PathAndQuery` extension method is a utility that helps construct URLs for redirection after updating the cart. This method takes into account both the path and query string of the current request.

:p What does the PathAndQuery extension method do?
??x
The `PathAndQuery` extension method constructs a URL by combining the path and query string from an HTTP request. It ensures that any additional parameters (query strings) are included in the generated URL, allowing for proper redirection after updating the cart.

```csharp
namespace SportsStore.Infrastructure {
    public static class UrlExtensions {
        public static string PathAndQuery(this HttpRequest request) => 
            request.QueryString.HasValue ? $"{request.Path}{request.QueryString}" : request.Path.ToString();
    }
}
```

x??

---

