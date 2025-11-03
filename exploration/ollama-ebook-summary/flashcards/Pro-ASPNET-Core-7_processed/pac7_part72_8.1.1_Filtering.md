# Flashcards: Pro-ASPNET-Core-7_processed (Part 72)

**Starting Chapter:** 8.1.1 Filtering the product list

---

#### Adding Navigation Controls to SportsStore
Background context: In this section, we aim to enhance the SportsStore application by adding navigation between product categories. This involves modifying the `HomeController` and the view model classes.

:p What is the main objective of enhancing the `ProductsListViewModel` class?
??x
The main objective is to add a property called `CurrentCategory` to communicate which category is currently selected to the view, allowing for rendering the sidebar with highlighted categories. This supports filtering products based on their category.
??x

---

#### Filtering Product List by Category
Background context: We need to modify the `Index` action method in the `HomeController` to filter product lists based on the current category. This involves updating the LINQ query within the `Index` method.

:p How does the modified `Index` action method handle filtering products by category?
??x
The `Index` action method now accepts a `category` parameter, which is used to filter products. If `category` is not null, only products matching the provided category are selected. The updated LINQ query looks like this:
```csharp
Products = repository.Products
    .Where(p => category == null || p.Category == category)
    .OrderBy(p => p.ProductID)
    .Skip((productPage - 1) * PageSize)
    .Take(PageSize),
```
This ensures that the product list is filtered based on the selected category and paginated accordingly.
??x

---

#### Updating Unit Tests for Category Filtering
Background context: To ensure the changes to the `Index` action method work as expected, we need to update our unit tests. Specifically, we must pass null as the first parameter to the `Index` method in tests that rely on it.

:p How should the existing unit tests be updated to handle the new category parameter?
??x
The existing unit tests should be updated to pass null for the category argument when working with the controller's `Index` action. For instance, in the `Can_Use_Repository` test:
```csharp
[Fact]
public void Can_Use_Repository()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1"}, 
        new Product {ProductID = 2, Name = "P2"}
    }).AsQueryable<Product>());
    HomeController controller = new HomeController(mock.Object);
    // Act
    ProductsListViewModel result = 
        controller.Index(null)?.ViewData.Model as ProductsListViewModel ?? new();
    // Assert
    Product[] prodArray = result.Products.ToArray();
    Assert.True(prodArray.Length == 2);
    Assert.Equal("P1", prodArray[0].Name);
    Assert.Equal("P2", prodArray[1].Name);
}
```
This ensures that the tests still work correctly with the new parameter handling.
??x

---

#### Displaying Filtered Products in View
Background context: The view associated with the `Index` action method needs to display products filtered by category. This involves rendering the correct product list based on the current category.

:p How is the current category passed to the view from the controller?
??x
The current category is passed to the view through the `ProductsListViewModel`. Specifically, the `CurrentCategory` property of the `ProductsListViewModel` is set in the `Index` action method as follows:
```csharp
Products = repository.Products
    .Where(p => category == null || p.Category == category)
    .OrderBy(p => p.ProductID)
    .Skip((productPage - 1) * PageSize)
    .Take(PageSize),
CurrentCategory = category;
```
This ensures that the view can use this property to render the correct sidebar and filter products accordingly.
??x

---

#### Ensuring Correct Pagination After Filtering
Background context: The `Index` action method also needs to ensure that pagination works correctly with filtered product lists. This means calculating the total number of items based on the category filter.

:p How is the total number of items calculated for pagination after filtering?
??x
The total number of items for pagination is currently incorrectly calculated without taking the category filter into account. To fix this, you would need to adjust the `PagingInfo.TotalItems` property within the `Index` action method:
```csharp
PagingInfo = new PagingInfo 
{
    CurrentPage = productPage,
    ItemsPerPage = PageSize,
    TotalItems = repository.Products.Where(p => category == null || p.Category == category).Count()
};
```
This ensures that the pagination logic correctly reflects the filtered product count.
??x

---

#### Mock Repository Setup for Testing
Background context: This section provides an example of setting up a mock repository to test a specific method or action in a controller. A mock object is used to simulate the behavior of a real repository, which helps ensure that the controller functions as expected.

:p How can you set up a mock repository for testing in C#?
??x
To set up a mock repository, you use the Moq library (or similar) to create a mock object. This mock object allows you to define behaviors and expectations that the real repository should follow during the tests. In this example, we are setting up a mock `IStoreRepository` that returns a predefined array of `Product` objects.

```csharp
// Arrange - create the mock repository
Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
mock.Setup(m => m.Products).Returns((new Product[] {
    new Product {ProductID = 1, Name = "P1", Category = "Cat1"},
    new Product {ProductID = 2, Name = "P2", Category = "Cat2"},
    new Product {ProductID = 3, Name = "P3", Category = "Cat1"},
    new Product {ProductID = 4, Name = "P4", Category = "Cat2"},
    new Product {ProductID = 5, Name = "P5", Category = "Cat3"}
}).AsQueryable<Product>());
```

The `Setup` method configures the repository to return a specific array of products when the `Products` property is accessed. The `AsQueryable` method makes it possible to use LINQ queries on the returned mock data.

x??

---

#### URL Scheme Refinement
Background context: This section discusses improving the URL structure for a web application to make URLs more user-friendly and readable. The original URLs were not very intuitive, leading to something like `/?category=Soccer`, which is ugly and difficult to understand.

:p How does changing the routing configuration in `Program.cs` improve URL readability?
??x
By refining the routing configuration, we can create cleaner URLs that represent the intended navigation paths more clearly. For example:

```csharp
app.MapControllerRoute(
    "catpage", 
    "{category}/Page{productPage:int}", 
    new { Controller = "Home", action = "Index" });
```

This route definition tells ASP.NET Core to map any URL that has a category and an optional page number, like `/Soccer/Page2`, to the `Index` method of the `HomeController`.

Additionally, we can define other routes for different navigation scenarios:
- `/Page{productPage:int}`: Lists products on a specific page.
- `{category}`: Lists the first page of items from a specific category.
- `Products/Page{productPage}`: Directly maps to listing products on a specific page.

These configurations ensure that URLs are more semantic and user-friendly, making them easier for both users and search engines to understand and navigate.

x??

---

#### Explanation of Routing System
Background context: This section explains how the routing system in ASP.NET Core handles incoming requests from clients and generates outgoing URLs based on predefined route templates. Understanding this is crucial for developing robust web applications with clean URL structures.

:p What does the routing system do in ASP.NET Core?
??x
The routing system in ASP.NET Core manages two main tasks:
1. **Handling Incoming Requests**: It processes HTTP requests to determine which controller and action method should handle them.
2. **Generating Outgoing URLs**: It creates URLs based on route templates, ensuring that these URLs conform to a specified URL scheme.

For example, the following lines from `Program.cs` define several routes:

```csharp
app.MapControllerRoute(
    "catpage", 
    "{category}/Page{productPage:int}", 
    new { Controller = "Home", action = "Index" });

app.MapControllerRoute(
    "page",
    "Page{productPage:int}",
    new { Controller = "Home", action = "Index", productPage = 1 });

app.MapControllerRoute(
    "category",
    "{category}",
    new { Controller = "Home", action = "Index", productPage = 1 });

app.MapControllerRoute(
    "pagination",
    "Products/Page{productPage}",
    new { Controller = "Home", action = "Index", productPage = 1 });
```

Each route template corresponds to a specific URL structure and maps it to the appropriate controller action. This setup ensures that URLs like `/Soccer/Page2` or `Products/Page5` are correctly routed to the desired actions.

x??

---

#### Route Summary Table
Background context: The table summarizes how different routes map to specific URLs, providing clarity on navigation paths within an application.

:p What is the route summary table in Listing 8.1 used for?
??x
The route summary table (Table 8.1) serves as a reference guide for understanding which URL corresponds to what functionality within the web application. It outlines different URL patterns and their mappings to specific actions:

| Route | URL                         | Leads To                                                                 |
|-------|-----------------------------|--------------------------------------------------------------------------|
| catpage | `{category}/Page{productPage:int}` | Lists the specified page of items from a specific category, e.g., `/Soccer/Page2` lists soccer products on page 2. |
| page   | `Page{productPage:int}`      | Lists the specified page of all categories' products, e.g., `/Page3` lists all products on page 3.                     |
| category | `{category}`                | Lists the first page of items from a specific category, e.g., `/Soccer` lists soccer products on the first page.        |
| pagination | `Products/Page{productPage}`   | Directly maps to listing products on a specific page without specifying a category, e.g., `/Products/Page5` lists all products on page 5. |

This table helps developers and testers quickly understand how different URLs behave in the application.

x??

---

#### Prefixed Values in Tag Helpers
Background context: The provided code snippet demonstrates how to use a tag helper with prefixed values to generate complex URLs. This approach allows for dynamic URL generation while maintaining flexibility and consistency within the application.

:p What is the purpose of using the `page-url-` prefix attribute in the `div` element?
??x
The purpose of using the `page-url-` prefix attribute is to allow additional query parameters to be dynamically added to the generated URLs. This ensures that context-specific information, such as the current category, can influence URL generation without requiring modifications to the tag helper class.

For example:
```html
<div page-model="@Model.PagingInfo" 
     page-action="Index" 
     page-classes-enabled="true" 
     page-class="btn" 
     page-class-normal="btn-outline-dark" 
     page-class-selected="btn-primary" 
     page-url-category="@Model.CurrentCategory." 
     class="btn-group pull-right m-1"></div>
```
In this case, the `page-url-category` attribute is prefixed with `page-url-`, allowing its value to be collected and passed as a URL parameter.

x??

---

#### Tag Helper Property Decorating
Background context: The code snippet shows how the `PageLinkTagHelper` class decorates properties using the `HtmlAttributeName` attribute. This allows for dynamic URL generation by collecting parameters in a dictionary, which can then be used with the `IUrlHelper.Action` method to create the URLs.

:p How does decorating tag helper properties with `HtmlAttributeName` facilitate the collection of additional URL parameters?
??x
Decorating tag helper properties with `HtmlAttributeName` enables the automatic collection of attributes that have a common prefix into a single dictionary. This makes it easier to pass multiple values to the URL generation process without cluttering the tag helper class.

For example, in the `PageLinkTagHelper`, the `PageUrlValues` dictionary collects all attributes whose names start with `page-url-`. This allows you to add parameters like category filters directly on the element being processed by the tag helper.

```csharp
[HtmlAttributeName(DictionaryAttributePrefix = "page-url-")]
public Dictionary<string, object> PageUrlValues { get; set; } = new Dictionary<string, object>();
```
The `PageUrlValues` dictionary can then be used to build dynamic URLs using these parameters:
```csharp
tag.Attributes["href"] = urlHelper.Action(PageAction, PageUrlValues);
```

x??

---

#### URL Generation with Tag Helpers
Background context: The example demonstrates how to generate complex URLs by combining the functionality of `IUrlHelper` and tag helpers. This method ensures that all URLs in an application are consistent and can be dynamically generated based on contextual data.

:p How does using the `IUrlHelperFactory` allow for URL generation within a tag helper?
??x
Using the `IUrlHelperFactory` allows you to create instances of `IUrlHelper`, which provides methods for generating URLs. By injecting `IUrlHelperFactory` into the tag helper, you can get an instance of `IUrlHelper` from the view context and use its `Action` method to generate URLs.

The code snippet initializes the `urlHelper` as follows:
```csharp
IUrlHelper urlHelper = urlHelperFactory.GetUrlHelper(ViewContext);
```
Then, the URL is generated using the `Action` method with the desired action name and a dictionary of values:
```csharp
tag.Attributes["href"] = urlHelper.Action(PageAction, PageUrlValues);
```

x??

---

#### Dynamic URL Generation Example
Background context: The example illustrates how to generate URLs dynamically by passing additional parameters through prefixed attributes on the tag helper element.

:p What is the significance of adding `page-url-category` in the view?
??x
Adding `page-url-category` in the view allows you to include category-specific information in the URL generated by the tag helper. This ensures that pagination links preserve the current category filter when clicked, maintaining consistent and meaningful URLs.

For example:
```html
<div page-model="@Model.PagingInfo" 
     page-action="Index" 
     page-classes-enabled="true" 
     page-class="btn" 
     page-class-normal="btn-outline-dark" 
     page-class-selected="btn-primary" 
     page-url-category="@Model.CurrentCategory." 
     class="btn-group pull-right m-1"></div>
```
The `page-url-category` attribute is prefixed with `page-url-`, allowing its value to be captured and used in the URL generation process. This ensures that when a user clicks on a pagination link, the current category remains part of the URL.

x??

---

---
#### View Components in ASP.NET Core
Background context: In ASP.NET Core, view components are C# classes that provide small amounts of reusable application logic and can be used to display Razor partial views. They offer a way to abstract UI pieces into separate modules, enhancing reusability and testability.

:p What is the purpose of creating a ViewComponent in an ASP.NET Core project?
??x
The purpose of creating a ViewComponent is to provide a modular and reusable piece of user interface logic that can be easily integrated into various views. This allows for better organization and maintenance of UI elements, as well as making it easier to test these components independently.

:p How does one use the `Invoke` method in a ViewComponent?
??x
The `Invoke` method is called when the view component is used in a Razor view. Its return value (which can be any type) is inserted into the HTML sent to the browser. In the initial example, it simply returns a string but will eventually return HTML.

:p How are view components integrated into a layout file?
??x
View components can be integrated into layout files using tag helpers. In the shared layout (`_Layout.cshtml`), you specify the view component by including its name in a special tag helper. For example, `<vc:navigation-menu />` references the `NavigationMenuViewComponent`.

:p How is the `Invoke` method implemented for the `NavigationMenuViewComponent`?
??x
The `Invoke` method of the `NavigationMenuViewComponent` returns a simple string message to demonstrate its basic functionality:

```csharp
using Microsoft.AspNetCore.Mvc;

namespace SportsStore.Components
{
    public class NavigationMenuViewComponent : ViewComponent
    {
        public string Invoke()
        {
            return "Hello from the Nav View Component";
        }
    }
}
```

:p Where should a view component be placed in an ASP.NET Core project?
??x
A view component should be placed in a folder named `Components` inside the main application project. This is the conventional home for view components, ensuring they are easily discoverable and maintainable.

:x??
---

#### Adding Categories to NavigationMenuViewComponent.cs
Background context: In this section, we update the `NavigationMenuViewComponent` to generate a list of categories from the repository. This involves using LINQ for data manipulation and dependency injection for accessing the repository.

The constructor defined in listing 8.8 defines an `IStoreRepository` parameter. When ASP.NET Core needs to create an instance of the view component class, it will note the need to provide a value for this parameter and inspect the configuration in the `Program.cs` file to determine which implementation object should be used.

:p How is the `NavigationMenuViewComponent` constructor designed?
??x
The `NavigationMenuViewComponent` constructor takes an `IStoreRepository` as a parameter. This allows the view component to access data without knowing which repository implementation will be used, leveraging dependency injection. The constructor assigns this parameter to a private field named `repository`.

```csharp
public class NavigationMenuViewComponent : ViewComponent {
    private IStoreRepository repository;

    public NavigationMenuViewComponent(IStoreRepository repo) {
        repository = repo;
    }
}
```
x??

---

#### Generating Categories Using LINQ in Invoke Method
Background context: The `Invoke` method of the view component uses LINQ to select and order categories from the repository. This data is then passed as an argument to a Razor partial view, which renders the HTML for displaying the categories.

:p How does the `Invoke` method generate and return the list of categories?
??x
The `Invoke` method in the `NavigationMenuViewComponent` uses LINQ to process the product data from the repository. It selects distinct category names, orders them alphabetically, and passes this ordered set as a model to the Razor partial view.

```csharp
public IViewComponentResult Invoke() {
    return View(repository.Products
        .Select(x => x.Category)
        .Distinct()
        .OrderBy(x => x));
}
```
x??

---

#### Dependency Injection in View Components
Background context: The `NavigationMenuViewComponent` constructor uses dependency injection to access the repository. This approach allows the view component to be decoupled from the specific implementation of the repository, making the code more flexible and maintainable.

:p How does dependency injection work in the context of the `NavigationMenuViewComponent`?
??x
Dependency injection is used here to inject an instance of `IStoreRepository` into the constructor of `NavigationMenuViewComponent`. When ASP.NET Core creates an instance of this view component, it resolves the required `IStoreRepository` implementation from the dependency injection container defined in `Program.cs`.

```csharp
public class NavigationMenuViewComponent : ViewComponent {
    private IStoreRepository repository;

    public NavigationMenuViewComponent(IStoreRepository repo) {
        repository = repo;
    }
}
```
x??

---

#### Unit Testing NavigationMenuViewComponent
Background context: A unit test is provided to verify that the `NavigationMenuViewComponent` correctly generates a sorted, unique list of categories. The test uses Moq to simulate the repository and asserts the expected behavior.

:p What is the purpose of the unit test for `NavigationMenuViewComponent`?
??x
The purpose of the unit test for `NavigationMenuViewComponent` is to ensure that it produces a correctly formatted and filtered set of categories. Specifically, the test checks if the list is sorted alphabetically and contains no duplicate entries.

```csharp
[Fact]
public void Can_Select_Categories() {
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1", Category = "Apples"}, 
        new Product {ProductID = 2, Name = "P2", Category = "Apples"}, 
        new Product {ProductID = 3, Name = "P3", Category = "Plums"}, 
        new Product {ProductID = 4, Name = "P4", Category = "Oranges"} 
    }).AsQueryable<Product>());
    
    NavigationMenuViewComponent target = new NavigationMenuViewComponent(mock.Object);

    // Act
    string[] results = ((IEnumerable<string>)(target.Invoke() as ViewViewComponentResult)?.ViewData?.Model ?? Enumerable.Empty<string>().ToArray());

    // Assert
    var expected = new List<string> { "Apples", "Oranges", "Plums" };
    Assert.Equal(expected.OrderBy(x => x), results);
}
```
x??

---

---
#### Mock Repository Implementation for Categories
Background context: The implementation of a mock repository for categories is described, where the repository contains repeating and unordered categories. The objective is to ensure that duplicates are removed and alphabetical ordering is enforced.

:p How does the mock repository handle categories?
??x
The mock repository stores categories in a way that may contain duplicates and be out of order. To ensure proper functionality, duplicates need to be removed and the categories should be alphabetically ordered before returning them.
```
public class MockCategoryRepository : ICategoryRepository
{
    private readonly List<string> _categories = new List<string>
    {
        "Fruits", "Vegetables", "Fruits", "Dairy"
    };

    public IEnumerable<string> GetCategories()
    {
        var uniqueCategories = _categories.Distinct().OrderBy(x => x);
        return uniqueCategories;
    }
}
```
x?
---

---
#### Using View Components for Navigation
Background context: The text explains how to use view components in Razor views, specifically the `NavigationMenu` component, which generates navigation links based on categories. It uses a default view and searches specific locations.

:p How does the `Default.cshtml` file generate category navigation links?
??x
The `Default.cshtml` file generates navigation links by using a foreach loop to iterate over each string in the model (categories). Each iteration creates an anchor (`<a>`) tag with routing attributes that point to the appropriate action and controller, passing the current category as a route parameter.

```razor
@model IEnumerable<string>

<div class="d-grid gap-2">
    <a class="btn btn-outline-secondary"asp-action="Index" asp-controller="Home" asp-route-category="">
        Home
    </a>
    @foreach (string category in Model ?? Enumerable.Empty<string>()) {
        <a class="btn btn-outline-secondary"
           asp-action="Index" asp-controller="Home"
           asp-route-category="@category"
           asp-route-productPage="1">
            @category
        </a>
    }
</div>
```
x?
---

---
#### Handling Current Request Data with ViewComponents
Background context: This section explains how to use the `RouteData` property in a view component to determine the currently selected category. The objective is to provide visual feedback on which category is active.

:p How can you access current request data using RouteData in a view component?
??x
You can access current request data by using the `RouteData` property in a view component. This property provides information about how the request URL was handled by the routing system, allowing you to determine the currently selected category and provide visual feedback.

:p Example code for accessing `RouteData` in a view component.
??x
```csharp
public class NavigationMenuViewComponent : ViewComponent
{
    public async Task<IViewComponentResult> InvokeAsync()
    {
        var routeData = Context.RouteData.Values;
        string selectedCategory = (string)routeData["category"] ?? "";
        
        // Use the selected category to generate navigation links with appropriate styling.
        return View(selectedCategory);
    }
}
```
x?
---

#### Passing Data to Views Using ViewBag
Background context: In ASP.NET Core MVC, you can pass data from a controller or view component to a view using `ViewBag`. This is an unstructured object that allows dynamic property assignments and value manipulations. It's particularly useful for passing temporary data or non-structural data.
:p How does ViewBag help in passing data to views?
??x
ViewBag helps in dynamically assigning properties to it without needing to define them beforehand, making it flexible for quick data passing. For example, you can directly set `ViewBag.SelectedCategory` to pass the selected category from a view component or controller to the view.
```csharp
// Example of setting ViewBag in a view component
public class NavigationMenuViewComponent : ViewComponent {
    public IViewComponentResult Invoke() {
        string selectedCategory = RouteData?.Values["category"];
        ViewBag.SelectedCategory = selectedCategory;
        return View();
    }
}
```
x??

---
#### Using RouteData to Retrieve Categories
Background context: The `RouteData` property in ASP.NET Core MVC contains information about the current route, including parameters like categories. You can use these values directly within view components or controllers.
:p How does `RouteData` help in retrieving category information?
??x
The `RouteData.Values["category"]` provides access to the category parameter passed via routing. This allows you to dynamically set properties such as `ViewBag.SelectedCategory`, which can then be used in views for conditional rendering or styling based on the selected category.
```csharp
// Example of using RouteData in a view component
public class NavigationMenuViewComponent : ViewComponent {
    public IViewComponentResult Invoke() {
        string selectedCategory = RouteData?.Values["category"];
        ViewBag.SelectedCategory = selectedCategory;
        return View(repository.Products.Select(x => x.Category).Distinct().OrderBy(x => x));
    }
}
```
x??

---
#### Unit Testing a View Component
Background context: Unit testing is crucial for ensuring that view components behave as expected. In this case, you want to verify that the `NavigationMenuViewComponent` correctly sets and passes the selected category through `ViewBag`.
:p How can we unit test that a view component adds details of the selected category?
??x
To unit test the `NavigationMenuViewComponent`, you need to set up routing data, configure mock dependencies, and validate the values in the `ViewData`. This ensures that the component correctly handles and passes the necessary information.
```csharp
// Example of a unit test for NavigationMenuViewComponent
[Fact]
public void Indicates_Selected_Category() {
    string categoryToSelect = "Apples";
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns(new[] { 
        new Product { ProductID = 1, Name = "P1", Category = "Apples" },
        new Product { ProductID = 4, Name = "P2", Category = "Oranges" }
    }.AsQueryable<Product>());
    NavigationMenuViewComponent target = new NavigationMenuViewComponent(mock.Object);
    target.ViewComponentContext = new ViewComponentContext {
        ViewContext = new ViewContext {
            RouteData = new RouteData()
        }
    };
    target.RouteData.Values["category"] = categoryToSelect;
    
    string? result = (string?)target.Invoke() as ViewViewComponentResult?.ViewData?["SelectedCategory"];
    Assert.Equal(categoryToSelect, result);
}
```
x??

---
#### Styling Links Based on Selected Category
Background context: Once you have passed the selected category via `ViewBag`, you can use this information in your views to apply specific CSS classes to navigation links. This helps highlight the current category and improves user experience.
:p How do we style navigation links based on the selected category?
??x
You can use CSS classes dynamically generated by checking if a link's text matches the selected category stored in `ViewBag.SelectedCategory`. For example, you might add a class like "selected" to the active link.
```html
<!-- Example of a navigation menu snippet -->
<ul>
    @foreach (var category in Model) {
        <li><a href="@Url.Action("List", new { category = category })" 
               class="@(category == ViewBag.SelectedCategory ? "selected" : "")">@category</a></li>
    }
</ul>
```
x??

---

