# Flashcards: Pro-ASPNET-Core-7_processed (Part 22)

**Starting Chapter:** 7.4 Adding pagination

---

#### Adding Pagination to the Product List View
Background context: The application currently displays all products on a single page, which can be overwhelming for users. Implementing pagination allows breaking down the product list into smaller, manageable pages, enhancing user experience and performance.

The `Index` method in the `HomeController` now accepts an optional parameter to control pagination:
```csharp
public ViewResult Index(int productPage = 1)
    => View(repository.Products
        .OrderBy(p => p.ProductID)
        .Skip((productPage - 1) * PageSize)
        .Take(PageSize));
```
- **productPage**: The page number being requested. By default, it is set to `1`.
- **PageSize**: Number of products per page, set to 4 in this case.

:p How does the `Index` method handle pagination?
??x
The `Index` method uses LINQ to OrderBy product IDs and then skips a certain number of products before taking the specified `PageSize`. This allows it to display a subset of the total products based on the requested page.
```csharp
public ViewResult Index(int productPage = 1)
{
    return View(repository.Products
        .OrderBy(p => p.ProductID) // Orders products by ID
        .Skip((productPage - 1) * PageSize) // Skips (page-1)*PageSize items
        .Take(PageSize)); // Takes the next PageSize items
}
```
x??

---
#### Mocking Repository for Pagination Unit Test
Background context: To ensure pagination works correctly, we need to create a unit test. We use Moq to mock the `IStoreRepository` and then request specific pages from the controller.

The code snippet provided starts the setup for this unit test:
```csharp
[Fact]
public void Can_Paginate()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
}
```

:p How do you set up a unit test to verify pagination functionality?
??x
We use Moq to create a mock repository and arrange the setup so that we can request specific pages. This allows us to check if the correct subset of data is returned based on the page number.
```csharp
[Fact]
public void Can_Paginate()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    // Additional code needed here, such as setting up return values for `Products` method
}
```
x??

---
#### Displaying Currency with ToString("c")
Background context: The price of each product is displayed using the `ToString("c")` method. This converts numerical prices into currency format according to the current server's culture settings.

Example:
```csharp
Price.ToString("c");
```

:p How does the `ToString("c")` method help in displaying prices?
??x
The `ToString("c")` method formats the price as a string representing currency, taking into account the server's culture settings. For example, if the server is set to `en-US`, it will display `$1,002.30`, while for `en-GB`, it would be `£1,002.30`.
```csharp
var price = 1002.3m;
string formattedPrice = price.ToString("c");
```
x??

---

#### Mock Setup for Testing Pagination
Background context explaining how mock setups are used to test functionalities. The provided example uses Moq to return a predefined set of `Product` objects and verify that the correct products are returned when paginating through them.

:p What is the purpose of the setup in the provided code snippet?
??x
The purpose of the setup is to define what data the mock will provide when the `Products` property is accessed. In this case, it returns an array of predefined `Product` objects that can be used to test pagination logic without needing a real database or service.

```csharp
mock.Setup(m => m.Products).Returns((new Product[] {
    new Product {ProductID = 1, Name = "P1"},
    new Product {ProductID = 2, Name = "P2"},
    new Product {ProductID = 3, Name = "P3"},
    new Product {ProductID = 4, Name = "P4"},
    new Product {ProductID = 5, Name = "P5"}
}).AsQueryable<Product>());
```
x??

---

#### Page Links in Paging
Explanation on how to implement and display page links for better user navigation. The objective is to allow users to navigate through pages of products using a more intuitive interface.

:p How can you add page links for pagination in the view?
??x
To add page links, we need to create a tag helper that generates HTML markup for the navigation links. This involves creating a view model class `PagingInfo` to pass information about the total items, items per page, and current page to the view.

```csharp
namespace SportsStore.Models.ViewModels {
    public class PagingInfo {
        public int TotalItems { get; set; }
        public int ItemsPerPage { get; set; }
        public int CurrentPage { get; set; }
        public int TotalPages => (int)Math.Ceiling((decimal)TotalItems / ItemsPerPage);
    }
}
```

The tag helper, `PageLinkTagHelper`, will use this view model to generate the necessary HTML for page links.

```csharp
namespace SportsStore.Infrastructure {
    public class PageLinkTagHelper : TagHelper {
        // implementation details go here
    }
}
```
x??

---

#### Implementation of PagingInfo Class
Explanation and code sample on how to implement the `PagingInfo` view model class.

:p What is the purpose of the `PagingInfo` class in pagination?
??x
The `PagingInfo` class is used to pass information about pagination to the view, such as the total number of items, items per page, and the current page. This helps in dynamically generating links for navigating through pages.

```csharp
namespace SportsStore.Models.ViewModels {
    public class PagingInfo {
        public int TotalItems { get; set; }
        public int ItemsPerPage { get; set; }
        public int CurrentPage { get; set; }
        public int TotalPages => (int)Math.Ceiling((decimal)TotalItems / ItemsPerPage);
    }
}
```

The `TotalPages` property uses a formula to calculate the total number of pages based on the total items and items per page.

```csharp
public int TotalPages => (int)Math.Ceiling((decimal)TotalItems / ItemsPerPage);
```
x??

---

#### PageLinkTagHelper Class
Explanation and code sample for creating a tag helper that generates navigation links.

:p How does the `PageLinkTagHelper` class work?
??x
The `PageLinkTagHelper` class is responsible for generating HTML markup for page navigation links. It uses the `PagingInfo` view model to determine how many pages exist, what the current page is, and what range of pages should be displayed.

```csharp
namespace SportsStore.Infrastructure {
    public class PageLinkTagHelper : TagHelper {
        private const int MaxLinks = 4;

        public PageLinkTagHelper(IUrlHelperFactory urlHelperFactory) {
            UrlHelperFactory = urlHelperFactory;
        }

        [ViewContext]
        public ViewContext ViewContext { get; set; }

        [HtmlAttributeNotBound]
        [ViewContext]
        public IUrlHelper UrlHelper => UrlHelperFactory.GetUrlHelper(ViewContext);

        [TempData]
        public string PageClassName { get; set; }

        public int PageCount { get; set; } = 1;

        public int PageNumber { get; set; } = 1;

        public bool ShowEllipsis { get; set; } = false;

        public override void Process(TagHelperContext context, TagHelperOutput output) {
            // implementation details go here
        }
    }
}
```

The `Process` method is where the actual logic for generating links happens. It uses the `PageCount`, `PageNumber`, and other properties to generate appropriate navigation links.

```csharp
public override void Process(TagHelperContext context, TagHelperOutput output) {
    // logic to generate page links goes here
}
```
x??

---

#### Tag Helpers Overview
Tag helpers are a feature of ASP.NET Core that allow you to introduce C# logic into your views. They are particularly useful for generating HTML content dynamically based on model data and application state. The key idea is to separate presentation logic from business logic, enhancing maintainability and testability.

:p What are tag helpers in ASP.NET Core?
??x
Tag helpers are a feature of ASP.NET Core that allow you to integrate C# code directly into Razor views, enabling dynamic generation of HTML elements based on model data. They help keep view logic separate from business logic, making the application more modular and easier to test.
x??

---

#### PageLinkTagHelper Class Details
The `PageLinkTagHelper` class is responsible for generating pagination links within a `div` element in Razor views. It uses dependency injection to obtain an `IUrlHelperFactory`, which it then uses to generate URLs based on the current page and total number of pages.

:p What is the purpose of the `PageLinkTagHelper`?
??x
The `PageLinkTagHelper` class is used to generate pagination links within a Razor view. It dynamically creates `<a>` tags that represent different product pages, linking them through generated URLs.
x??

---

#### Implementing PageLinkTagHelper
To implement the `PageLinkTagHelper`, you need to define it in a specific namespace and mark it with an attribute specifying which HTML element it can work on.

:p How do you register the `PageLinkTagHelper`?
??x
You register the `PageLinkTagHelper` by adding a statement to the `_ViewImports.cshtml` file within the Views folder. This tells ASP.NET Core where to find the tag helper classes and makes it available for use in views.
```csharp
@addTagHelper *, SportsStore
```
x??

---

#### Using Tag Helpers in Views
To use `PageLinkTagHelper` in a Razor view, you include it within an HTML element with specific attributes. The `Process` method of the tag helper generates the necessary HTML content.

:p How do you use `PageLinkTagHelper` in a Razor view?
??x
You can use `PageLinkTagHelper` by including it within a `div` element and providing attributes such as `page-model`, `page-action`, and `page-model`. The tag helper will then generate the necessary pagination links.
```html
<div page-model="@(Model.PagingInfo)" page-action="Test">
</div>
```
x??

---

#### Testing Tag Helpers
Testing tag helpers involves creating a test setup where you can provide mocked dependencies and inspect the output generated by the tag helper.

:p How do you unit test `PageLinkTagHelper`?
??x
To unit test `PageLinkTagHelper`, you mock the necessary dependencies such as `IUrlHelperFactory` and use them to create instances of the tag helper. You then call its `Process` method with test data and inspect the output to ensure it generates the correct HTML.
```csharp
[Fact]
public void Can_Generate_Page_Links()
{
    // Arrange
    var urlHelper = new Mock<IUrlHelper>();
    // Setup mock behavior for URL generation
    // Act
    PageLinkTagHelper helper = new PageLinkTagHelper(urlHelper.Object)
    {
        PageModel = new PagingInfo { CurrentPage = 2, TotalItems = 28, ItemsPerPage = 10 },
        ViewContext = viewContext.Object,
        PageAction = "Test"
    };
    TagHelperContext ctx = new TagHelperContext(
        new TagHelperAttributeList(),
        new Dictionary<object, object>(),
        "");
    var content = new Mock<TagHelperContent>();
    TagHelperOutput output = new TagHelperOutput("div",
        new TagHelperAttributeList(),
        (cache, encoder) => Task.FromResult(content.Object));
    helper.Process(ctx, output);
    // Assert
    Assert.Equal(@"<a href=""Test/Page1"">1</a>
<a href=""Test/Page2"">2</a>
<a href=""Test/Page3"">3</a>", 
        output.Content.GetContent());
}
```
x??

---

#### Using Literal Strings with Double Quotes in C#
Background context: In C#, when working with strings that contain double quotes, you can use a literal string by prefixing it with @ and using two sets of double quotes (\"") instead of one. This is necessary for ensuring the string does not break.
If applicable, add code examples with explanations:
```csharp
string example = @"This is a ""test"" string.";
```
:p How do you handle strings containing double quotes in C#?
??x
You use a verbatim string literal by prefixing the string with @ and using two sets of double quotes (\"") instead of one. This ensures that the string is interpreted correctly without breaking.
```csharp
string example = @"This is a ""test"" string.";
```
x??

---

#### Creating the ProductsListViewModel Class
Background context: The `ProductsListViewModel` class is created to provide product and pagination information to the view. It uses properties to store these details, making it easier to pass data from the controller to the view.
If applicable, add code examples with explanations:
```csharp
namespace SportsStore.Models.ViewModels {
    public class ProductsListViewModel {
        public IEnumerable<Product> Products { get; set; } = Enumerable.Empty<Product>();
        public PagingInfo PagingInfo { get; set; } = new();
    }
}
```
:p What is the purpose of the `ProductsListViewModel` class?
??x
The `ProductsListViewModel` class is used to pass product data and pagination details from the controller to the view. It encapsulates a collection of products and paging information, making it easier to manage and display paginated data in the view.
```csharp
namespace SportsStore.Models.ViewModels {
    public class ProductsListViewModel {
        public IEnumerable<Product> Products { get; set; } = Enumerable.Empty<Product>();
        public PagingInfo PagingInfo { get; set; } = new();
    }
}
```
x??

---

#### Updating the Index Action Method in HomeController
Background context: The `Index` action method is updated to use an instance of `ProductsListViewModel` as a model. This allows the controller to provide both product and pagination data to the view.
If applicable, add code examples with explanations:
```csharp
public class HomeController : Controller {
    private IStoreRepository repository;
    public int PageSize = 4;

    public HomeController(IStoreRepository repo) { 
        repository = repo; 
    }

    public ViewResult Index(int productPage = 1) => 
        View(new ProductsListViewModel {
            Products = repository.Products
                .OrderBy(p => p.ProductID)
                .Skip((productPage - 1) * PageSize)
                .Take(PageSize),
            PagingInfo = new PagingInfo {
                CurrentPage = productPage,
                ItemsPerPage = PageSize,
                TotalItems = repository.Products.Count()
            }
        });
}
```
:p How does the `Index` action method update to use `ProductsListViewModel`?
??x
The `Index` action method is updated to create an instance of `ProductsListViewModel`, which contains a collection of products and pagination information. The view data for the model is then set to this `ProductsListViewModel` object, allowing it to pass both product and pagination details to the view.
```csharp
public class HomeController : Controller {
    private IStoreRepository repository;
    public int PageSize = 4;

    public HomeController(IStoreRepository repo) { 
        repository = repo; 
    }

    public ViewResult Index(int productPage = 1) => 
        View(new ProductsListViewModel {
            Products = repository.Products
                .OrderBy(p => p.ProductID)
                .Skip((productPage - 1) * PageSize)
                .Take(PageSize),
            PagingInfo = new PagingInfo {
                CurrentPage = productPage,
                ItemsPerPage = PageSize,
                TotalItems = repository.Products.Count()
            }
        });
}
```
x??

---

#### Unit Testing the Controller
Background context: A unit test is created to ensure that the controller sends the correct pagination data to the view. This involves setting up a mock repository and verifying that the `ProductsListViewModel` object contains the expected product and pagination information.
If applicable, add code examples with explanations:
```csharp
[Fact]
public void Can_Send_Pagination_View_Model() {
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1"}, 
        new Product {ProductID = 2, Name = "P2"}, 
        new Product {ProductID = 3, Name = "P3"}, 
        new Product {ProductID = 4, Name = "P4"}, 
        new Product {ProductID = 5, Name = "P5"} 
    }).AsQueryable<Product>());
    
    // Arrange
    HomeController controller = 
        new HomeController(mock.Object) { PageSize = 3 };
    
    // Act
    ProductsListViewModel result = 
        controller.Index(2)?.ViewData.Model as ProductsListViewModel 
        ?? null;
}
```
:p How does the unit test ensure correct pagination data is sent to the view?
??x
The unit test ensures that the `ProductsListViewModel` object contains the expected product and pagination information by setting up a mock repository. It then calls the `Index` action method with page 2, verifies that it returns a non-null `ProductsListViewModel`, and checks that its properties match the expected values.
```csharp
[Fact]
public void Can_Send_Pagination_View_Model() {
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1"}, 
        new Product {ProductID = 2, Name = "P2"}, 
        new Product {ProductID = 3, Name = "P3"}, 
        new Product {ProductID = 4, Name = "P4"}, 
        new Product {ProductID = 5, Name = "P5"} 
    }).AsQueryable<Product>());
    
    // Arrange
    HomeController controller = 
        new HomeController(mock.Object) { PageSize = 3 };
    
    // Act
    ProductsListViewModel result = 
        controller.Index(2)?.ViewData.Model as ProductsListViewModel 
        ?? null;
}
```
x??

---

#### Mocking and Testing Setup
Background context: The provided text describes setting up unit tests for a controller action that deals with paginated products. This involves using mocking frameworks to simulate repository behavior and testing different scenarios such as fetching all products or paginating them.

:p How is the `Can_Use_Repository` test setup?
??x
The test sets up a mock repository to return predefined product data, then invokes the controller's Index action with this setup. It asserts that the returned model contains the expected number and names of products.
```csharp
[Fact]
public void Can_Use_Repository()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1"}, 
        new Product {ProductID = 2, Name = "P2" }
    }).AsQueryable<Product>());
    HomeController controller = new HomeController(mock.Object);

    // Act
    ProductsListViewModel result = 
        controller.Index()?.ViewData.Model as ProductsListViewModel ?? new();
    
    // Assert
    Product[] prodArray = result.Products.ToArray();
    Assert.True(prodArray.Length == 2);
    Assert.Equal("P1", prodArray[0].Name);
    Assert.Equal("P2", prodArray[1].Name);
}
```
x??

---
#### Paginating Products Test
Background context: The text also describes testing the pagination feature of the product list. This involves setting up a mock repository with more products and then invoking the Index action method with specific page parameters to test pagination logic.

:p How is the `Can_Paginate` test setup?
??x
The test sets up a mock repository with multiple products, configures the controller's page size property, invokes the Index action with a specific page number, and asserts that the returned model contains the correct subset of products.
```csharp
[Fact]
public void Can_Paginate()
{
    // Arrange
    Mock<IStoreRepository> mock = new Mock<IStoreRepository>();
    mock.Setup(m => m.Products).Returns((new Product[] { 
        new Product {ProductID = 1, Name = "P1"}, 
        new Product {ProductID = 2, Name = "P2"}, 
        new Product {ProductID = 3, Name = "P3"}, 
        new Product {ProductID = 4, Name = "P4"}, 
        new Product {ProductID = 5, Name = "P5"} 
    }).AsQueryable<Product>());
    HomeController controller = new HomeController(mock.Object);
    controller.PageSize = 3;

    // Act
    ProductsListViewModel result = 
        controller.Index(2)?.ViewData.Model as ProductsListViewModel ?? new();

    // Assert
    Product[] prodArray = result.Products.ToArray();
    Assert.True(prodArray.Length == 2);
    Assert.Equal("P4", prodArray[0].Name);
    Assert.Equal("P5", prodArray[1].Name);
}
```
x??

---
#### ViewModel Update for Index Action
Background context: The text mentions updating the view model and the corresponding view to handle paginated product data. This involves changing the `ProductsListViewModel` class and updating the `Index.cshtml` file to display products using this new structure.

:p What changes were made to the `Index.cshtml` file?
??x
The `Index.cshtml` file was updated to use the `ProductsListViewModel` type for its model. The foreach loop was modified to iterate over the `Products` property of the model, and it includes logic to display each product's name, description, and price.
```csharp
@model ProductsListViewModel

@foreach (var p in Model.Products ?? Enumerable.Empty<Product>()) {
    <div>
        <h3>@p.Name</h3>
        @p.Description
        <h4>@p.Price.ToString("c")</h4>
    </div>
}
```
x??

---

#### Adding Pagination to the View
To add pagination to the view, the model type was updated to `ProductsListViewModel`, and an additional HTML element with a specific attribute is added. The `div` tag helper processes this attribute to generate page links for navigating through paginated data.

:p How does the `div` tag helper create pagination links in the view?
??x
The `div` tag helper uses the provided `page-model` attribute and the information within it (such as `PagingInfo`) to generate a set of navigation links. It transforms the `<div>` element into HTML containing hyperlinks that allow users to navigate through different pages.

```html
<div page-model="@Model.PagingInfo" page-action="Index"></div>
```

x??

---

#### Updating the Model View Type
The `@model` directive was changed from a simpler type to `ProductsListViewModel`, which contains information about the products and paging details. This change is necessary for handling paginated data properly.

:p Why did you change the model view type from a simpler type to `ProductsListViewModel`?
??x
By changing the model view type to `ProductsListViewModel`, we can include both product information and paging details in the model passed to the view. This allows us to handle pagination more effectively, as it provides all necessary data for generating page links.

```csharp
@model ProductsListViewModel
```

x??

---

#### Adding Page Navigation Links
To display the navigation links, an HTML `div` element was added with a specific attribute. The tag helper processes this attribute and generates the links based on the provided paging information.

:p What is the role of the `page-model` attribute in generating page navigation links?
??x
The `page-model` attribute instructs the PageLinkTagHelper to generate a set of navigation links for paginated data. When Razor encounters this attribute, it processes the model and generates HTML links that allow users to navigate through different pages.

```html
<div page-model="@Model.PagingInfo" page-action="Index"></div>
```

x??

---

#### Changing URL Scheme for Pagination
The query string parameter `productPage` was replaced with a more user-friendly URL scheme by adding a route in the `Program.cs` file. This change improves the URLs and makes them more appealing to users.

:p How did you modify the URL scheme for pagination?
??x
To change the URL scheme, a new route was added in the `Program.cs` file that uses the pattern `Products/Page{productPage}`. The application routes these URLs to the `Index` action of the `Home` controller, allowing navigation without query strings.

```csharp
app.MapControllerRoute("pagination",
                        "Products/Page{productPage}",
                        new { Controller = "Home", action = "Index" });
```

x??

---

#### Generating User-Friendly URLs with ASP.NET Core Routing
By adding a new route in the `Program.cs` file, the application can generate more user-friendly URLs for navigating through paginated data. This improves the URL structure and enhances the user experience.

:p What is the benefit of using composable URL patterns like "http://localhost/Page2" instead of query string URLs?
??x
Using composable URL patterns such as "http://localhost/Page2" improves readability and makes URLs more memorable for users. It separates page navigation from other query parameters, making the URLs cleaner and easier to understand.

```csharp
app.MapControllerRoute("pagination",
                        "Products/Page{productPage}",
                        new { Controller = "Home", action = "Index" });
```

x??

---

#### Ensuring URL Changes Reflect in the Application
After adding a new route for pagination, ASP.NET Core automatically updates the URLs used by the application. Tag helpers like `page-model` are aware of these changes and generate appropriate links.

:p How does ASP.NET Core routing ensure that the new URL scheme is applied throughout the application?
??x
ASP.NET Core routing integrates tightly with tag helpers such as `page-model`. When the route configuration is updated, any URLs generated by tag helpers will automatically reflect the new structure. This ensures consistent and user-friendly navigation across the application.

```csharp
app.MapControllerRoute("pagination",
                        "Products/Page{productPage}",
                        new { Controller = "Home", action = "Index" });
```

x??

---

