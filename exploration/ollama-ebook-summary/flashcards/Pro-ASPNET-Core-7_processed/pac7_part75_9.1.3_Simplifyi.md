# Flashcards: Pro-ASPNET-Core-7_processed (Part 75)

**Starting Chapter:** 9.1.3 Simplifying the cart Razor Page

---

---
#### Creating a Service for Cart Class
Background context: The text explains how to create and use a service for managing cart objects within an ASP.NET Core application. The goal is to seamlessly store cart data using session data, ensuring that changes made to the cart persist across HTTP requests.

:p How does the `AddScoped` method work in this context?
??x
The `AddScoped` method ensures that the same object instance of `Cart` (specifically `SessionCart`) is reused for related requests within a single HTTP request. This means any component handling the same HTTP request will receive the same `Cart` object, allowing session data to be stored and retrieved seamlessly.

```csharp
builder.Services.AddScoped<Cart>(sp => SessionCart.GetCart(sp));
```
x?
---

---
#### Singleton Service for HttpContextAccessor
Background context: A singleton service is registered with ASP.NET Core to provide access to the current HTTP context. This service is crucial because it allows the `SessionCart` class to access the current session data, which is necessary for storing and retrieving cart items.

:p Why is a singleton service used for `HttpContextAccessor`?
??x
A singleton service is used for `HttpContextAccessor` because it ensures that only one instance of `HttpContext` is shared across all components in the application. This shared instance provides access to session data, cookies, and other HTTP context-related information needed by `SessionCart`.

```csharp
builder.Services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();
```
x?
---

---
#### Simplifying Cart Razor Page Model
Background context: The text discusses how the cart Razor page model can be simplified by using a service to manage `Cart` objects. This approach allows the page model to focus on its role in the application without handling session storage explicitly.

:p How does using a constructor argument for `Cart` simplify the page model class?
??x
Using a constructor argument for `Cart` simplifies the page model class by abstracting away the complexities of loading and storing sessions. Instead, the class can directly use the provided `Cart` object, making the code cleaner and easier to understand.

```csharp
public class CartModel : PageModel {
    private IStoreRepository repository;
    public CartModel(IStoreRepository repo, Cart cartService) {
        repository = repo;
        Cart = cartService;
    }
    public Cart Cart { get; set; }
}
```
x?
---

---
#### Updating Unit Tests for Cart Model
Background context: The text explains how the unit tests for the `Cart` model need to be updated when the service is introduced. This ensures that test cases remain valid and accurate after refactoring.

:p How should the unit tests be modified for reading and updating the cart?
??x
The unit tests should modify their approach to creating and managing the `Cart` object by providing it as a constructor argument rather than accessing session data directly. This aligns with the new service-based approach, ensuring that the tests accurately reflect the application's behavior.

For example:

```csharp
[Fact]
public void Can_Load_Cart() {
    // Arrange
    Product p1 = new Product { ProductID = 1, Name = "P1" };
    Product p2 = new Product { ProductID = 2, Name = "P2" };
    Mock<IStoreRepository> mockRepo = new Mock<IStoreRepository>();
    mockRepo.Setup(m => m.Products).Returns((new Product[] {
        p1, p2
    }).AsQueryable<Product>());

    // - create a cart
    Cart testCart = new Cart();
    testCart.AddItem(p1, 2);
    testCart.AddItem(p2, 1);

    // Action
    CartModel cartModel = new CartModel(mockRepo.Object, testCart);
    cartModel.OnGet("myUrl");

    // Assert
    Assert.Equal(2, cartModel.Cart.Lines.Count());
    Assert.Equal("myUrl", cartModel.ReturnUrl);
}
```
x?
---

---
#### Adding a Remove Button to Cart.cshtml
The context is adding functionality to allow users to remove items from their cart on the Cart Razor Page. This involves updating both the HTML and C# code to handle HTTP POST requests.

:p How does the Remove button in Cart.cshtml work?
??x
The Remove button triggers an HTTP POST request that calls a handler method within the `CartModel` class. When a user clicks the Remove button, it submits a form containing hidden fields for the product ID and return URL. The handler method processes this request to remove the specified item from the cart.

Relevant code in Cart.cshtml:
```html
<form asp-page-handler="Remove" method="post">
    <input type="hidden" name="ProductID" value="@line.Product.ProductID" />
    <input type="hidden" name="returnUrl" value="@Model?.ReturnUrl" />
    <button type="submit" class="btn btn-sm btn-danger">Remove</button>
</form>
```
The `asp-page-handler` attribute specifies which handler method in the page model should handle this request. In this case, it's `OnPostRemove`.

:p What is the logic of the OnPostRemove method in Cart.cshtml.cs?
??x
The `OnPostRemove` method in the `CartModel` class removes a specific item from the cart based on the product ID received from the POST request.

```csharp
public IActionResult OnPostRemove(long productId, string returnUrl)
{
    Cart.RemoveLine(Cart.Lines.First(cl => cl.Product.ProductID == productId).Product);
    return RedirectToPage(new { returnUrl = returnUrl });
}
```
This method:
1. Receives `productId` and `returnUrl` as parameters.
2. Finds the product in the cart using `Cart.Lines`.
3. Removes that product from the cart with `Cart.RemoveLine`.
4. Redirects to the specified URL (`returnUrl`) after removal.

The logic ensures that when a user clicks "Remove," the corresponding item is deleted from their cart, and they are redirected back to the previous page.
x??
---

#### Adding a Cart Summary Widget to SportsStore

Background context: The goal is to enhance the user experience by providing a persistent cart summary widget that shows the contents of the shopping cart at all times, rather than forcing users to navigate to a separate cart summary screen.

:p What is the purpose of adding a cart summary widget?
??x
The purpose is to allow customers to see what items are in their cart and provide an easy way to proceed to checkout without needing to visit a dedicated cart summary page. This improves user experience by making it more convenient for users to manage their orders.
x??

---

#### Installing the Font Awesome Package

Background context: To display a visually appealing "checkout" button, the Font Awesome package is used because it provides icons that can be embedded as characters in web pages.

:p How do you install the Font Awesome package in the SportsStore project?
??x
To install the Font Awesome package, use a PowerShell command prompt to run the following command:
```powershell
libman install font-awesome@6.2.1 -d wwwroot/lib/font-awesome
```
This command installs version 6.2.1 of the Font Awesome package in the `wwwroot/lib/font-awesome` directory.
x??

---

#### Creating a View Component for Cart Summary

Background context: A view component is created to display a summary of the cart, including the number of items and their total value.

:p What does the `CartSummaryViewComponent.cs` class do?
??x
The `CartSummaryViewComponent.cs` class is a custom view component that takes a `Cart` object as a constructor argument. It passes this cart to the view method to generate HTML content for displaying the cart summary.

```csharp
using Microsoft.AspNetCore.Mvc;
using SportsStore.Models;

namespace SportsStore.Components
{
    public class CartSummaryViewComponent : ViewComponent
    {
        private readonly Cart _cartService;

        public CartSummaryViewComponent(Cart cartService)
        {
            _cartService = cartService;
        }

        public IViewComponentResult Invoke()
        {
            return View(_cartService);
        }
    }
}
```
x??

---

#### Creating the Razor View for the Cart Summary

Background context: A Razor view is created to display the summary of the cart in a user-friendly manner, including the number of items and their total value.

:p What does the `Default.cshtml` file in the `Views/Shared/Components/CartSummary` folder contain?
??x
The `Default.cshtml` file contains HTML code that displays the cart summary. If there are items in the cart, it shows a snapshot detailing the number of items and their total value, along with a checkout button using Font Awesome icons.

```html
@model Cart

<div class="">
    @if (Model.Lines.Count() > 0) {
        <small class="navbar-text">
            <b>Your cart:</b>
            @Model.Lines.Sum(x => x.Quantity) item(s)
            @Model.ComputeTotalValue().ToString("c")
        </small>
    }
    <a class="btn btn-sm btn-secondary navbar-btn" asp-page="/Cart"
       asp-route-returnurl="@ViewContext.HttpContext.Request.PathAndQuery()">
        <i class="fa fa-shopping-cart"></i>
    </a>
</div>
```
x??

---

#### Integrating the Cart Summary Widget in Layout

Background context: The cart summary widget is integrated into the `_Layout.cshtml` file to be displayed on all pages.

:p How does the integration of the cart summary widget look in the layout?
??x
The integration of the cart summary widget in the layout adds a navigation component that displays the number of items in the cart and their total value, along with a checkout button. The code snippet below shows how it is integrated:

```html
<!DOCTYPE html>
<html>
<head>
    <meta name="viewport" content="width=device-width" />
    <title>SportsStore</title>
    <link href="/lib/bootstrap/css/bootstrap.min.css" rel="stylesheet" />
    <link href="/lib/font-awesome/css/all.min.css" rel="stylesheet" />
</head>
<body>
    <div class="bg-dark text-white p-2">
        <div class="container-fluid">
            <div class="row">
                <div class="col navbar-brand">SPORTS STORE</div>
                <div class="col-6 navbar-text text-end">
                    <vc:cart-summary />
                </div>
            </div>
        </div>
    </div>
    <div class="row m-1 p-1">
        <div id="categories" class="col-3">
            <vc:navigation-menu />
        </div>
        <div class="col-9">
            @RenderBody()
        </div>
    </div>
</body>
</html>
```
x??

---

