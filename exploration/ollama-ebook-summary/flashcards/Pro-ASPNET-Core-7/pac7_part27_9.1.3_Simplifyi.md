# Flashcards: Pro-ASPNET-Core-7_processed (Part 27)

**Starting Chapter:** 9.1.3 Simplifying the cart Razor Page

---

---
#### Adding Services for Cart Management
Background context: In this section, we are focusing on adding a service to manage `Cart` objects within an e-commerce application. The goal is to use session storage to maintain cart information across user interactions. This involves setting up services that handle `Cart` requests using ASP.NET Core's dependency injection system.

:p How can you ensure that the same `Cart` object is used for related requests in a web application?
??x
To ensure that the same `Cart` object is used for related requests, we use the `AddScoped` method to add the `Cart` service. This method tells ASP.NET Core to create an instance of `Cart` per HTTP request and share it among all components handling that request.

```csharp
builder.Services.AddScoped<Cart>(sp => SessionCart.GetCart(sp));
```

Here, a lambda expression is used to return a `SessionCart` object, which will store itself in the session when modified. This ensures that any page or controller that needs a cart within the same HTTP request gets the same instance.

x??
---
#### HttpContextAccessor for Session Management
Background context: To access the current session within the `SessionCart` class, we need to use an `HttpContextAccessor`. This service is added using the `AddSingleton` method, ensuring that the same `HttpContextAccessor` object is used throughout the application.

:p Why is it necessary to add a `HttpContextAccessor` as a singleton service?
??x
It is necessary to add a `HttpContextAccessor` as a singleton service because this class provides access to the current HTTP context, which includes session data. Without this accessor, we would not be able to read or write session-based cart information from within the `SessionCart` class.

```csharp
builder.Services.AddSingleton<IHttpContextAccessor, HttpContextAccessor>();
```

This service is required for the `SessionCart.GetCart()` method to work correctly and retrieve the current session data.

x??
---
#### Simplifying Cart Razor Page Code
Background context: The goal here is to simplify the code where `Cart` objects are used by leveraging services. By injecting a `Cart` object into the page model constructor, we can remove the need for session management logic from handler methods, making the code cleaner and more maintainable.

:p How does the introduction of a service for managing `Cart` objects simplify the Razor Page?
??x
The introduction of a service for managing `Cart` objects simplifies the Razor Page by eliminating the need to explicitly load or store sessions in the handler methods. Instead, we can use dependency injection to get the `Cart` object directly.

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

By doing this, the `Cart` object is automatically managed by the service and stored in the session. This makes the code more focused on its primary responsibilities without worrying about how carts are created or persisted.

x??
---
#### Refactoring Unit Tests for Cart Model
Background context: To ensure that the new service approach works correctly, we need to update our unit tests. Specifically, the test setup must provide a `Cart` object as a constructor argument instead of trying to retrieve it from the session within the test methods.

:p How should the unit tests be modified when using the service for managing `Cart` objects?
??x
When using the service for managing `Cart` objects, the unit tests need to modify their setup by providing the `Cart` object as a constructor argument. This ensures that the test code directly uses the provided cart instead of attempting to load it from session data.

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

This approach ensures that the tests are isolated and focused on the behavior of the `Cart` model without relying on external session storage.

x??
---

#### Testing Cart Update Functionality

Background context explaining how to test the cart update functionality. This involves setting up a mock repository, initializing a cart, and verifying that the cart is updated correctly.

:p How can you test whether an item has been added to the cart using a mock setup?
??x
To test adding an item to the cart, we first create a mock repository with a single product. Then, we initialize a `Cart` object and use it to instantiate a `CartModel`. The `OnPost` method of the model is called with specific parameters (product ID and return URL). Finally, assertions are made to check that the cart contains the expected item.

```csharp
// Arrange
Mock<IStoreRepository> mockRepo = new Mock<IStoreRepository>();
mockRepo.Setup(m => m.Products).Returns((new Product[] {
    new Product { ProductID = 1, Name = "P1" }
}).AsQueryable<Product>());

Cart testCart = new Cart();

// Action
CartModel cartModel = new CartModel(mockRepo.Object, testCart);
cartModel.OnPost(1, "myUrl");

// Assert
Assert.Single(testCart.Lines);
Assert.Equal("P1", testCart.Lines.First().Product.Name);
Assert.Equal(1, testCart.Lines.First().Quantity);
```
x??

---

#### Removing Items from the Cart

Background context explaining how to add a Remove button to allow customers to remove items from their cart. This involves modifying the Razor page and adding a handler method in the page model.

:p How can you add a Remove button in the Razor Page to delete an item from the cart?
??x
To add a Remove button, HTML forms are used within each row of the cart table. Each form submits a POST request when the "Remove" button is clicked. The form includes hidden fields for `ProductID` and `returnUrl`. The button triggers the submission by setting its `type` to `submit`.

```html
<tr>
    <td class="text-center">@line.Quantity</td>
    <td class="text-left">@line.Product.Name</td>
    <td class="text-right">@line.Product.Price.ToString("c")</td>
    <td class="text-right">@(line.Quantity * line.Product.Price).ToString("c")</td>
    <td class="text-center">
        <form asp-page-handler="Remove" method="post">
            <input type="hidden" name="ProductID" value="@line.Product.ProductID" />
            <input type="hidden" name="returnUrl" value="@Model?.ReturnUrl" />
            <button type="submit" class="btn btn-sm btn-danger">Remove</button>
        </form>
    </td>
</tr>
```
x??

---

#### Removing an Item from the Cart in the Page Model

Background context explaining how to handle the removal of items from the cart within the page model. This involves defining a new handler method that processes the request and modifies the cart accordingly.

:p How does the `CartModel` class handle removing items when the "Remove" button is clicked?
??x
The `CartModel` class includes an `OnPostRemove` method to handle the removal of items from the cart. This method receives parameters such as `productId` and `returnUrl`. It locates the item in the cart using the provided product ID and removes it.

```csharp
public IActionResult OnPostRemove(long productId, string returnUrl)
{
    Cart.RemoveLine(Cart.Lines.First(cl => cl.Product.ProductID == productId).Product);
    return RedirectToPage(new { returnUrl = returnUrl });
}
```
x??

---

#### Summary of Cart Functionality

Background context explaining the completion of cart functionality by adding new features such as removing items and displaying a summary.

:p What are the two new features added to the cart functionality in this section?
??x
The two new features added to the cart functionality are:
1. **Removing Items from the Cart**: A "Remove" button is added to each item in the cart, allowing customers to delete items by submitting an HTTP POST request.
2. **Displaying a Summary of the Cart**: The total value of the cart is displayed at the bottom of the page, providing a summary to the customer.

These features enhance user experience and ensure that customers can manage their carts effectively.
x??

---

#### Adding a Cart Summary Widget
Background context: The cart functionality is currently functional but has an integration issue. Customers can only view their cart contents by adding new items, which is inconvenient and unintuitive. To improve user experience, we are implementing a widget that summarizes the cart content and provides direct access to it throughout the application.

:p What problem does the current cart summary implementation have?
??x
The current cart summary is only accessible after adding an item to the cart. This makes it hard for customers to know what items they currently have in their cart without navigating away from the product pages.
x??

---

#### Installing Font Awesome Package
Background context: To make the checkout button more user-friendly, we are using a Font Awesome icon instead of text. We will install the package and use its icons throughout the application.

:p What is the purpose of installing the Font Awesome package?
??x
The purpose is to enable the use of visually appealing icons in our application, specifically for displaying the cart symbol on the checkout button.
x??

---

#### Creating CartSummaryViewComponent Class
Background context: We need a view component that will summarize the contents of the cart and can be included in various parts of the application. This component will receive a `Cart` object as an argument and pass it to the view for rendering.

:p How does the `CartSummaryViewComponent` class work?
??x
The `CartSummaryViewComponent` class takes a `Cart` object as a constructor argument, which allows it to access the cart contents. It then passes this `Cart` object to the `Invoke()` method, which returns a view component result containing the summarized cart information.

```csharp
using Microsoft.AspNetCore.Mvc;
using SportsStore.Models;

namespace SportsStore.Components {
    public class CartSummaryViewComponent : ViewComponent {
        private readonly Cart _cart;

        public CartSummaryViewComponent(Cart cart) {
            _cart = cart;
        }

        public IViewComponentResult Invoke() {
            return View(_cart);
        }
    }
}
```
x??

---

#### Creating Default.cshtml File
Background context: The `Default.cshtml` file defines the view that will be used by the `CartSummaryViewComponent`. This view will display the number of items in the cart and their total value, along with a checkout button.

:p What does the `Default.cshtml` file do?
??x
The `Default.cshtml` file is responsible for rendering the summary information about the cart. It checks if there are any items in the cart, then displays the total number of items and their combined cost. Additionally, it includes a checkout button with a Font Awesome icon.

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

#### Modifying the Layout to Include Cart Summary
Background context: To ensure that the cart summary is always available, we need to modify the layout file (`_Layout.cshtml`) so it includes our `CartSummaryViewComponent` in a prominent location.

:p How does modifying the `_Layout.cshtml` file include the cart summary?
??x
We add the `vc:cart-summary` tag helper to the navigation bar within the layout file. This tag helper invokes the `CartSummaryViewComponent`, which renders the cart summary view component's output into the layout at runtime.

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
        <div class="col-9">@RenderBody()</div>
    </div>
</body>
</html>
```
x??

---

